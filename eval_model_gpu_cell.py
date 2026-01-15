import sys
import os
import time
import json
import numpy as np
import cv2
import traceback
import matplotlib
matplotlib.use('Qt5Agg')  # 指定 Qt5 後端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns 

import tensorflow as tf
from skimage import measure

# --- MRCNN Imports ---
import mrcnn.model as modellib
from mrcnn.utils import compute_ap, compute_matches
from mrcnn.config import Config

# --- Custom Dataset ---
# 🔥 確保你的資料夾內有修正後的 CustomCroppingDataset.py (含 np.round)
from CustomCroppingDataset import CustomCroppingDataset
from CustomDataset import CustomDataset

# --- PyQt5 Imports ---
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QFileDialog, QLabel, QLineEdit, QCheckBox, 
                             QTextEdit, QHBoxLayout, QProgressBar, QMessageBox,
                             QTabWidget, QGridLayout, QScrollArea, QFrame, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap

# 防止 GPU OOM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# =================================================================================
# Config (完全同步 Training Release 1.0)
# =================================================================================
class InferenceConfig(Config):
    NAME = "cell"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2
    BACKBONE = "resnet101"
    
    # 🔥 [關鍵] 網子夠大
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_ANCHOR_RATIOS = [0.75, 1, 1.33, 1.6]
    
    # 🔥 [關鍵] 使用 COCO 標準 (配合 Training)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    
    # 🔥 [關鍵] 關閉 Mini Mask (銳利邊緣)
    USE_MINI_MASK = False
    
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_MAX_INSTANCES = 100
    DETECTION_NMS_THRESHOLD = 0.3
    
    # 🔥 [關鍵] 鎖定 512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_RESIZE_MODE = "pad64"

# =================================================================================
# 核心邏輯：指標與繪圖
# =================================================================================

def compute_metrics_at_threshold(gt_boxes, gt_class_ids, gt_masks,
                                 pred_boxes, pred_class_ids, pred_scores, pred_masks,
                                 iou_threshold=0.5, ignore_fp=False):
    """
    計算單一 IoU 門檻下的 AP, Precision, Recall
    """
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold=iou_threshold)

    # Compute AP
    ap, precisions, recalls, _ = compute_ap(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold=iou_threshold)

    tp = np.sum(pred_match > -1)
    fp = np.sum(pred_match == -1)
    fn = np.sum(gt_match == -1)

    if ignore_fp:
        fp = 0 

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return ap, precision, recall, gt_match, pred_match, overlaps

def draw_color_coded_result(image, gt_mask, pred_mask, pred_class_ids, pred_match):
    """
    繪製：🟨TP 🟥FP(Cell) 🟦FP(Chromo) 🟩GT
    """
    canvas = image.copy()
    
    # GT (Green Outline)
    if gt_mask.shape[-1] > 0:
        for i in range(gt_mask.shape[-1]):
            contours = measure.find_contours(gt_mask[..., i], 0.5)
            for contour in contours:
                rr, cc = contour[:, 0].astype(int), contour[:, 1].astype(int)
                rr = np.clip(rr, 0, canvas.shape[0]-1)
                cc = np.clip(cc, 0, canvas.shape[1]-1)
                canvas[rr, cc] = [0, 255, 0] 

    # Predictions
    if pred_mask.shape[-1] > 0:
        for i in range(pred_mask.shape[-1]):
            class_id = pred_class_ids[i]
            is_tp = pred_match[i] > -1
            
            if is_tp:
                color = [255, 255, 0] # Yellow
            else:
                color = [255, 0, 0] if class_id == 1 else [0, 0, 255] # Red / Blue
            
            mask = pred_mask[..., i]
            contours = measure.find_contours(mask, 0.5)
            for contour in contours:
                rr, cc = contour[:, 0].astype(int), contour[:, 1].astype(int)
                rr = np.clip(rr, 0, canvas.shape[0]-1)
                cc = np.clip(cc, 0, canvas.shape[1]-1)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                         r_off = np.clip(rr + dx, 0, canvas.shape[0]-1)
                         c_off = np.clip(cc + dy, 0, canvas.shape[1]-1)
                         canvas[r_off, c_off] = color
    return canvas

# =================================================================================
# Worker Thread
# =================================================================================
class EvaluationThread(QThread):
    progress_signal = pyqtSignal(int, int)
    status_signal = pyqtSignal(str)
    result_image_signal = pyqtSignal(np.ndarray, str) 
    finished_signal = pyqtSignal(dict) 
    error_signal = pyqtSignal(str)

    def __init__(self, dataset_path, weights_path, output_folder, limit, use_cpu, ignore_fp):
        super().__init__()
        self.dataset_path = dataset_path
        self.weights_path = weights_path
        self.output_folder = output_folder
        self.limit = limit
        self.use_cpu = use_cpu
        self.ignore_fp = ignore_fp

    def run(self):
        try:
            import ray
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)

            if self.use_cpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

            config = InferenceConfig()
            model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.path.dirname(self.weights_path))
            model.load_weights(self.weights_path, by_name=True)

            dataset = CustomDataset()
            subset = "test" if os.path.exists(os.path.join(self.dataset_path, "test")) else "train"
            dataset.load_custom(self.dataset_path, subset)
            dataset.prepare()

            limit = len(dataset.image_ids) if self.limit == -1 else min(self.limit, len(dataset.image_ids))
            
            # --- Metrics Containers ---
            precisions_05 = []
            recalls_05 = []
            ious_list = []
            cm_data = [] 
            
            # Multi-threshold Metrics (for mAP vs IoU Graph)
            iou_thresholds = np.arange(0.5, 1.0, 0.05)
            map_curve_data = {th: [] for th in iou_thresholds} 

            start_time = time.time()

            for i in range(limit):
                image_id = dataset.image_ids[i]
                
                # Load GT & Infer
                image, _, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id)
                results = model.detect([image], verbose=0)
                r = results[0]

                # --- A. Main Calculation (IoU=0.5) ---
                ap_05, p_05, rec_05, gt_match, pred_match, overlaps = compute_metrics_at_threshold(
                    gt_bbox, gt_class_id, gt_mask,
                    r['rois'], r['class_ids'], r['scores'], r['masks'],
                    iou_threshold=0.5, ignore_fp=self.ignore_fp
                )
                
                precisions_05.append(p_05)
                recalls_05.append(rec_05)
                if overlaps.size > 0: ious_list.extend(np.max(overlaps, axis=1))

                # CM Data (修正變數名稱 gt_class_id)
                for idx, pred_idx in enumerate(np.where(pred_match > -1)[0]):
                    gt_idx = int(pred_match[pred_idx])
                    cm_data.append((gt_class_id[gt_idx], r['class_ids'][pred_idx]))
                
                for idx in np.where(gt_match == -1)[0]:
                    cm_data.append((gt_class_id[idx], 0))
                
                if not self.ignore_fp:
                    for idx in np.where(pred_match == -1)[0]:
                        cm_data.append((0, r['class_ids'][idx]))

                # --- B. Multi-threshold Calculation (Loop 0.5 -> 0.95) ---
                for th in iou_thresholds:
                    ap, _, _, _ = compute_ap(
                        gt_bbox, gt_class_id, gt_mask,
                        r['rois'], r['class_ids'], r['scores'], r['masks'],
                        iou_threshold=th
                    )
                    map_curve_data[th].append(ap)

                # Draw Result
                vis_img = draw_color_coded_result(image, gt_mask, r['masks'], r['class_ids'], pred_match)
                
                # Save to disk if output_folder is set
                if self.output_folder:
                    save_path = os.path.join(self.output_folder, f"eval_{i:03d}.png")
                    try:
                        # Convert RGB to BGR for OpenCV saving
                        cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                    except Exception as ex:
                        print(f"Error saving image: {ex}")

                # Emit to GUI
                self.result_image_signal.emit(vis_img, f"Img {i}")
                
                elapsed = time.time() - start_time
                fps = (i+1) / elapsed
                self.status_signal.emit(f"Processing {i+1}/{limit} | FPS: {fps:.1f} | Recall(0.5): {rec_05:.2f}")
                self.progress_signal.emit(i+1, limit)

            map_curve_y = [np.mean(map_curve_data[th]) for th in iou_thresholds]

            stats = {
                "precisions": precisions_05,
                "recalls": recalls_05,
                "ious": ious_list,
                "cm_data": cm_data,
                "mAP_05": np.mean(precisions_05),
                "curve_x": iou_thresholds,
                "curve_y": map_curve_y
            }
            self.finished_signal.emit(stats)

        except Exception as e:
            self.error_signal.emit(str(e) + "\n" + traceback.format_exc())
# =================================================================================
# GUI Tabs
# =================================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class EvaluationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell R-CNN Evaluation Dashboard (Ultimate V3)")
        self.resize(1300, 900)
        
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        self.tab_control = QWidget()
        self.init_control_tab()
        
        self.tab_gallery = QWidget()
        self.init_gallery_tab()

        self.tab_analytics = QWidget()
        self.init_analytics_tab()

        self.tabs.addTab(self.tab_control, "🎛️ Control")
        self.tabs.addTab(self.tab_gallery, "🖼️ Visual Results")
        self.tabs.addTab(self.tab_analytics, "📊 Analytics")
        
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)
        self.auto_load_default()

    def init_control_tab(self):
        layout = QVBoxLayout()
        
        # Profile
        profile_layout = QHBoxLayout()
        self.save_btn = QPushButton("💾 Save Profile As...")
        self.load_btn = QPushButton("📂 Load Profile...")
        profile_layout.addWidget(self.save_btn)
        profile_layout.addWidget(self.load_btn)
        layout.addLayout(profile_layout)
        
        # Inputs
        self.dataset_input = QLineEdit()
        self.weights_input = QLineEdit()
        self.output_input = QLineEdit() # Output Path
        
        self.dataset_btn = QPushButton("Browse Dataset")
        self.weights_btn = QPushButton("Browse Weights")
        self.output_btn = QPushButton("Browse Output")
        
        # Settings
        self.limit_input = QLineEdit("50")
        self.cpu_chk = QCheckBox("Force CPU")
        self.ignore_fp_chk = QCheckBox("⚠️ Ignore False Positives (Incomplete GT Mode)")
        self.ignore_fp_chk.setChecked(True) 
        self.ignore_fp_chk.setStyleSheet("color: red; font-weight: bold;")
        
        self.run_btn = QPushButton("🚀 Start Evaluation")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; padding: 10px; font-weight: bold;")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        self.progress = QProgressBar()
        
        # Rows
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Dataset Path:")); h1.addWidget(self.dataset_input); h1.addWidget(self.dataset_btn)
        
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Weights File:")); h2.addWidget(self.weights_input); h2.addWidget(self.weights_btn)
        
        h_out = QHBoxLayout() # Output Row
        h_out.addWidget(QLabel("Output Folder:")); h_out.addWidget(self.output_input); h_out.addWidget(self.output_btn)
        
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Image Limit:")); h3.addWidget(self.limit_input)
        h3.addWidget(self.cpu_chk); h3.addWidget(self.ignore_fp_chk)
        
        layout.addLayout(h1); layout.addLayout(h2); layout.addLayout(h_out); layout.addLayout(h3)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.progress)
        layout.addWidget(self.log_text)
        self.tab_control.setLayout(layout)
        
        # Connections
        self.dataset_btn.clicked.connect(lambda: self._browse(self.dataset_input, True))
        self.weights_btn.clicked.connect(lambda: self._browse(self.weights_input, False))
        self.output_btn.clicked.connect(lambda: self._browse(self.output_input, True))
        
        self.run_btn.clicked.connect(self.start_eval)
        self.save_btn.clicked.connect(self.save_profile_dialog)
        self.load_btn.clicked.connect(self.load_profile_dialog)

    def init_gallery_tab(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.gallery_content = QWidget()
        self.gallery_grid = QGridLayout()
        self.gallery_content.setLayout(self.gallery_grid)
        scroll.setWidget(self.gallery_content)
        
        legend = QLabel("Legend: 🟩 Green Outline = GT | 🟨 Yellow = Correct (TP) | 🟥 Red = Cell FP | 🟦 Blue = Chromo FP")
        legend.setStyleSheet("font-weight: bold; background-color: #e0e0e0; padding: 8px; border-radius: 4px;")
        layout.addWidget(legend)
        layout.addWidget(scroll)
        self.tab_gallery.setLayout(layout)
        self.gallery_idx = 0

    def init_analytics_tab(self):
        main_layout = QVBoxLayout()
        grid_widget = QWidget()
        layout = QGridLayout()
        
        self.canvas_pr = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas_cm = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas_iou = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas_map_curve = MplCanvas(self, width=5, height=4, dpi=100)
        
        layout.addWidget(QLabel("📈 Recall / Precision (Avg)"), 0, 0)
        layout.addWidget(self.canvas_pr, 1, 0)
        layout.addWidget(QLabel("📊 Confusion Matrix"), 0, 1)
        layout.addWidget(self.canvas_cm, 1, 1)
        layout.addWidget(QLabel("🎯 IoU Distribution"), 2, 0)
        layout.addWidget(self.canvas_iou, 3, 0)
        layout.addWidget(QLabel("📉 mAP vs. IoU Threshold"), 2, 1)
        layout.addWidget(self.canvas_map_curve, 3, 1)
        
        grid_widget.setLayout(layout)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #f0f0f0; padding: 10px;")
        self.stats_text.setText("Waiting for results...")
        
        main_layout.addWidget(grid_widget)
        main_layout.addWidget(self.stats_text)
        self.tab_analytics.setLayout(main_layout)

    def _browse(self, line, is_dir):
        if is_dir: path = QFileDialog.getExistingDirectory(self)
        else: path, _ = QFileDialog.getOpenFileName(self, "Select Weights", filter="Weights (*.h5)")
        if path: line.setText(path)

    def save_profile_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Profile", "profile_eval.json", "JSON Files (*.json)")
        if not path: return
        profile = {
            'dataset': self.dataset_input.text(),
            'weights': self.weights_input.text(),
            'output': self.output_input.text(), # 🔥 Saved output path
            'limit': self.limit_input.text(),
            'use_cpu': self.cpu_chk.isChecked(),
            'ignore_fp': self.ignore_fp_chk.isChecked()
        }
        try:
            with open(path, "w") as f: json.dump(profile, f)
            self.log_text.append(f"💾 Profile saved to {os.path.basename(path)}")
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def load_profile_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Profile", "", "JSON Files (*.json)")
        if not path: return
        self._load_from_path(path)

    def auto_load_default(self):
        if os.path.exists("profile_eval_dashboard.json"): self._load_from_path("profile_eval_dashboard.json")
        elif os.path.exists("profile_eval.json"): self._load_from_path("profile_eval.json")

    def _load_from_path(self, path):
        try:
            with open(path) as f: profile = json.load(f)
            self.dataset_input.setText(profile.get('dataset', ''))
            self.weights_input.setText(profile.get('weights', ''))
            self.output_input.setText(profile.get('output', '')) # 🔥 Load output path
            self.limit_input.setText(profile.get('limit', '50'))
            self.cpu_chk.setChecked(profile.get('use_cpu', False))
            self.ignore_fp_chk.setChecked(profile.get('ignore_fp', True))
            self.log_text.append(f"📂 Loaded: {os.path.basename(path)}")
        except Exception as e: self.log_text.append(f"❌ Error loading {path}: {e}")

    def start_eval(self):
        self.gallery_idx = 0
        for i in reversed(range(self.gallery_grid.count())): 
            self.gallery_grid.itemAt(i).widget().setParent(None)
        
        dataset_path = self.dataset_input.text()
        weights_path = self.weights_input.text()
        output_path = self.output_input.text()
        
        if not os.path.exists(dataset_path) or not os.path.exists(weights_path):
            QMessageBox.warning(self, "Path Error", "Please check dataset or weights path.")
            return
        
        # Create output folder if specified
        if output_path and not os.path.exists(output_path):
            try: os.makedirs(output_path)
            except: pass

        self.run_btn.setEnabled(False)
        self.log_text.append("🚀 Starting Evaluation...")
        try: limit = int(self.limit_input.text())
        except: limit = 50
        
        self.thread = EvaluationThread(dataset_path, weights_path, output_path, limit, self.cpu_chk.isChecked(), self.ignore_fp_chk.isChecked())
        self.thread.status_signal.connect(lambda s: self.log_text.append(s))
        self.thread.progress_signal.connect(self.progress.setValue)
        self.thread.result_image_signal.connect(self.add_image_to_gallery)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.error_signal.connect(lambda s: QMessageBox.critical(self, "Error", s))
        self.thread.start()

    def add_image_to_gallery(self, image, title):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label = QLabel()
        label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
        label.setToolTip(title)
        label.setFrameStyle(QFrame.Box)
        row = self.gallery_idx // 4
        col = self.gallery_idx % 4
        self.gallery_grid.addWidget(label, row, col)
        self.gallery_idx += 1

    def on_finished(self, stats):
        self.run_btn.setEnabled(True)
        self.log_text.append("✅ Evaluation Complete!")
        self.update_graphs(stats)
        self.tabs.setCurrentIndex(2) 

    def update_graphs(self, stats):
        txt = f"<b>Total Images:</b> {len(stats['precisions'])}<br>"
        txt += f"<b>Mean Recall (IoU 0.5):</b> {np.mean(stats['recalls']):.4f}<br>"
        txt += f"<b>Mean Precision (IoU 0.5):</b> {np.mean(stats['precisions']):.4f}<br>"
        txt += f"<b>Mean IoU:</b> {np.mean(stats['ious']):.4f}<br>"
        txt += f"<b>mAP @ IoU 0.5:</b> {stats['mAP_05']:.4f}<br>"
        if self.ignore_fp_chk.isChecked():
            txt += "<br><font color='red'>⚠️ [Mode: Ignore False Positives] Precision is calculated based on GT detection only.</font>"
        self.stats_text.setHtml(txt)
        
        self.canvas_iou.axes.clear()
        self.canvas_iou.axes.hist(stats['ious'], bins=20, color='#ff9800', edgecolor='black', alpha=0.7)
        self.canvas_iou.axes.set_title("IoU Distribution")
        self.canvas_iou.axes.set_xlabel("IoU Score")
        self.canvas_iou.draw()
        
        self.canvas_pr.axes.clear()
        self.canvas_pr.axes.plot(stats['recalls'], label='Recall', color='#2196f3', linewidth=2)
        self.canvas_pr.axes.plot(stats['precisions'], label='Precision', color='#4caf50', linestyle='--', linewidth=2)
        self.canvas_pr.axes.legend()
        self.canvas_pr.axes.set_title("Recall & Precision (IoU 0.5)")
        self.canvas_pr.axes.grid(True, linestyle=':', alpha=0.6)
        self.canvas_pr.draw()
        
        self.canvas_cm.axes.clear()
        cm_data = stats['cm_data']
        if cm_data:
            cm = np.zeros((3, 3), dtype=int)
            for t, p in cm_data:
                if t < 3 and p < 3: cm[t, p] += 1
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=self.canvas_cm.axes,
                        xticklabels=['BG', 'Cell', 'Chromo'], yticklabels=['BG', 'Cell', 'Chromo'])
            self.canvas_cm.axes.set_ylabel('True Label')
            self.canvas_cm.axes.set_xlabel('Pred Label')
            self.canvas_cm.draw()

        self.canvas_map_curve.axes.clear()
        x = stats['curve_x']
        y = stats['curve_y']
        
        self.canvas_map_curve.axes.plot(x, y, marker='o', linestyle='-', color='#9c27b0', linewidth=2)
        self.canvas_map_curve.axes.set_title("mAP vs. IoU Threshold")
        self.canvas_map_curve.axes.set_xlabel("IoU Threshold")
        self.canvas_map_curve.axes.set_ylabel("mAP")
        self.canvas_map_curve.axes.set_ylim(0, 1.05)
        self.canvas_map_curve.axes.grid(True, linestyle='--', alpha=0.7)
        self.canvas_map_curve.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = EvaluationApp()
    gui.show()
    sys.exit(app.exec_())
