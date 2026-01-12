import sys
import os
import time
import json
import numpy as np
import matplotlib
# Use 'Agg' backend to prevent Matplotlib/Qt conflicts
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from skimage import measure

# Redirect standard output/error to capture logs
from io import StringIO

import mrcnn.model as modellib
from mrcnn.utils import compute_ap
from mrcnn.config import Config

# Import your datasets
from CustomCroppingDataset import CustomCroppingDataset
from CustomDataset import CustomDataset

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QFileDialog, QLabel, QLineEdit, QCheckBox, 
                             QTextEdit, QHBoxLayout, QProgressBar, QMessageBox, QStatusBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

# --- FIX START: Enhanced Stream Redirector ---
class StreamRedirector(QObject):
    """Redirects stdout/stderr to a Qt signal, while keeping file attributes for libraries like Ray."""
    text_written = pyqtSignal(str)

    def __init__(self, original_stream=None):
        super().__init__()
        self.original_stream = original_stream

    def write(self, text):
        self.text_written.emit(str(text))
        # Optional: Uncomment if you still want logs in the real console/terminal
        # if self.original_stream:
        #     self.original_stream.write(text)
        #     self.original_stream.flush()

    def flush(self):
        if self.original_stream:
            self.original_stream.flush()

    def fileno(self):
        """Required by Ray/subprocess to access the file descriptor."""
        if self.original_stream:
            try:
                return self.original_stream.fileno()
            except (AttributeError, ValueError):
                pass
        return 1  # Default to stdout fd if unknown

    def isatty(self):
        """Required by some logging libraries."""
        if self.original_stream:
            try:
                return self.original_stream.isatty()
            except AttributeError:
                pass
        return False
# --- FIX END ---

# --- Configuration ---
class InferenceConfig(Config):
    NAME = "cell"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    # Match Training Config (Critical for good results)
    IMAGE_MIN_DIM = 1024 
    IMAGE_MAX_DIM = 1024 
    IMAGE_RESIZE_MODE = "pad64" 
    
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    BACKBONE = "resnet101"
    NUM_CLASSES = 1 + 2  # Background + Cell + Chromosome
    USE_MINI_MASK = False
    
    DETECTION_MIN_CONFIDENCE = 0.7 
    DETECTION_NMS_THRESHOLD = 0.3
    RPN_NMS_THRESHOLD = 0.7
    DETECTION_MAX_INSTANCES = 50

# --- Helper Function ---
def unmold_mask(mask, bbox, image_shape):
    y1, x1, y2, x2 = bbox
    mask = cv2.resize(mask.astype(np.float32), (x2 - x1, y2 - y1))
    mask = np.where(mask >= 0.5, 1, 0).astype(np.bool_)
    full_mask = np.zeros(image_shape[:2], dtype=np.bool_)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask

# --- Worker Thread for Evaluation ---
class EvaluationThread(QThread):
    progress_signal = pyqtSignal(int, int) # Current, Total
    status_signal = pyqtSignal(str)        # For Status Bar (ETA, mAP)
    finished_signal = pyqtSignal(str)      # Final Report
    error_signal = pyqtSignal(str)         # Critical Errors

    def __init__(self, dataset_path, weights_path, output_folder, limit, use_cpu):
        super().__init__()
        self.dataset_path = dataset_path
        self.weights_path = weights_path
        self.output_folder = output_folder
        self.limit = limit
        self.use_cpu = use_cpu

    def run(self):
        try:
            # Ensure Ray is initialized safely
            import ray
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)

            # 1. Setup Environment
            if self.use_cpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                print("⚙️ Mode: CPU")
            else:
                print("⚙️ Mode: GPU")

            # 2. Load Configuration
            config = InferenceConfig()
            
            # 3. Load Dataset with Progress Bar Connection
            print(f"📂 Loading dataset from: {self.dataset_path}")
            dataset = CustomDataset()
            
            # --- CALLBACK FUNCTION FOR LOADING ---
            def loading_callback(current, total):
                # Emit signal to update GUI Progress Bar
                self.progress_signal.emit(current, total)
                # Emit signal to update Status Text
                self.status_signal.emit(f"Loading Data: {current}/{total}")
            # -------------------------------------

            # Pass the callback to load_custom
            dataset.load_custom(self.dataset_path, "train", progress_callback=loading_callback) 
            
            dataset.prepare()
            print(f"✅ Dataset loaded. Total images: {len(dataset.image_ids)}")

            # 4. Load Model
            print(f"⚖️ Loading weights: {os.path.basename(self.weights_path)}")
            model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.path.dirname(self.weights_path))
            model.load_weights(self.weights_path, by_name=True)

            # 5. Determine limit
            total_images = len(dataset.image_ids)
            target_limit = total_images if (self.limit == -1 or self.limit > total_images) else self.limit

            precisions = []
            class_precisions = {1: [], 2: []}
            
            print(f"🚀 Starting evaluation on {target_limit} images...")
            
            start_time = time.time()

            # 6. Main Evaluation Loop
            for i in range(target_limit):
                image_id = dataset.image_ids[i]
                
                # Load image and GT
                try:
                    image, _, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id)
                except Exception as e:
                    print(f"⚠️ Error loading image {image_id}: {e}")
                    continue

                if gt_mask.size == 0:
                    continue

                # Run Inference
                molded_image = np.expand_dims(modellib.mold_image(image.astype(np.float32), config), 0)
                results = model.detect(molded_image, verbose=0)
                r = results[0]

                # Resize masks
                masks_resized = [unmold_mask(r["masks"][:, :, j], r["rois"][j], image.shape) for j in range(r["masks"].shape[-1])]
                r["masks"] = np.stack(masks_resized, axis=-1) if masks_resized else np.zeros(image.shape[:2] + (0,))

                # Compute AP
                AP, P, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r["masks"], iou_threshold=0.5)
                precisions.append(AP)

                for k, class_id in enumerate(r["class_ids"]):
                    class_precisions.setdefault(class_id, []).append(P[k])

                # Stats
                current_mean_ap = np.mean(precisions)
                elapsed = time.time() - start_time
                avg_time_per_img = elapsed / (i + 1)
                remaining_imgs = target_limit - (i + 1)
                eta_seconds = int(remaining_imgs * avg_time_per_img)
                eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
                
                status_msg = f"ETA: {eta_str} | mAP: {current_mean_ap:.4f} | Current AP: {AP:.4f}"
                
                self.status_signal.emit(status_msg)
                self.progress_signal.emit(i + 1, target_limit)

                # Save Plot
                self.save_visualization(image, gt_mask, r['masks'], i)

            # 7. Final Report
            final_mean_ap = np.mean(precisions) if precisions else 0
            report = [
                f"\n🏁 Evaluation Complete!",
                f"📊 Final Mean AP: {final_mean_ap:.4f}"
            ]
            for cid, p in class_precisions.items():
                if p:
                    report.append(f"   - Class {cid} AP: {np.mean(p):.4f}")
            
            final_msg = "\n".join(report)
            print(final_msg)
            self.finished_signal.emit(final_msg)

        except Exception as e:
            self.error_signal.emit(str(e))
            import traceback
            traceback.print_exc()

    def save_visualization(self, image, gt_mask, pred_mask, index):
        # ... (same as before) ...
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.imshow(image)
        for j in range(gt_mask.shape[-1]):
            for contour in measure.find_contours(gt_mask[..., j], 0.5):
                ax.plot(contour[:, 1], contour[:, 0], '-g', linewidth=1)
        for j in range(pred_mask.shape[-1]):
            for contour in measure.find_contours(pred_mask[..., j], 0.5):
                ax.plot(contour[:, 1], contour[:, 0], '-r', linewidth=1)
        ax.axis('off')
        save_path = os.path.join(self.output_folder, f'image_{index:03d}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
# --- Main GUI Class ---
class EvaluationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mask R-CNN Evaluation GUI")
        self.resize(750, 650)
        self.thread = None

        # --- FIX: Pass original streams to redirector ---
        # We need to save the original stdout/stderr first
        self.original_stdout = sys.__stdout__
        self.original_stderr = sys.__stderr__

        self.stdout_redirector = StreamRedirector(self.original_stdout)
        self.stderr_redirector = StreamRedirector(self.original_stderr)
        
        self.stdout_redirector.text_written.connect(self.append_log)
        self.stderr_redirector.text_written.connect(self.append_log)
        
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stderr_redirector

        # UI Components
        self.dataset_input = QLineEdit()
        self.weights_input = QLineEdit()
        self.output_input = QLineEdit()
        self.limit_field = QLineEdit("100")
        self.cpu_checkbox = QCheckBox("Use CPU")
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")

        self.dataset_btn = QPushButton("Browse")
        self.weights_btn = QPushButton("Browse")
        self.output_btn = QPushButton("Browse")
        self.eval_btn = QPushButton("Run Evaluation")
        self.eval_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        self.save_btn = QPushButton("Save Profile")
        self.load_btn = QPushButton("Load Profile")

        # --- Status Bar Components ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("QProgressBar { text-align: center; }")
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold; padding-left: 5px;")

        # Layout Setup
        layout = QVBoxLayout()
        
        # Input Rows
        for lbl, line, btn in [("Dataset:", self.dataset_input, self.dataset_btn),
                               ("Weights:", self.weights_input, self.weights_btn),
                               ("Output:", self.output_input, self.output_btn)]:
            row = QHBoxLayout()
            label = QLabel(lbl)
            label.setFixedWidth(60)
            row.addWidget(label)
            row.addWidget(line)
            row.addWidget(btn)
            layout.addLayout(row)

        # Settings
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Limit:"))
        self.limit_field.setFixedWidth(80)
        settings_row.addWidget(self.limit_field)
        settings_row.addWidget(self.cpu_checkbox)
        settings_row.addStretch()
        layout.addLayout(settings_row)

        layout.addWidget(self.eval_btn)
        
        # Log Section
        layout.addWidget(QLabel("Console Output:"))
        layout.addWidget(self.result_text)
        
        # Profile Buttons
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.load_btn)
        layout.addLayout(btn_row)

        # --- Status Bar Area (Bottom) ---
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_label, 1) # Label takes available space
        status_layout.addWidget(self.progress_bar, 2) # Bar takes twice the space
        layout.addLayout(status_layout)

        self.setLayout(layout)

        # Connections
        self.dataset_btn.clicked.connect(lambda: self._browse(self.dataset_input, True))
        self.weights_btn.clicked.connect(lambda: self._browse(self.weights_input, False, "Weights (*.h5)"))
        self.output_btn.clicked.connect(lambda: self._browse(self.output_input, True))
        self.eval_btn.clicked.connect(self.start_evaluation)
        self.save_btn.clicked.connect(self.save_profile)
        self.load_btn.clicked.connect(self.load_profile)

    def _browse(self, lineedit, is_folder, file_filter=None):
        if is_folder:
            path = QFileDialog.getExistingDirectory(self, "Select Directory")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", filter=file_filter)
        if path:
            lineedit.setText(path)

    def save_profile(self):
        profile = {
            'dataset': self.dataset_input.text(),
            'weights': self.weights_input.text(),
            'output': self.output_input.text(),
            'limit': self.limit_field.text(),
            'use_cpu': self.cpu_checkbox.isChecked()
        }
        with open("profile_eval.json", "w") as f:
            json.dump(profile, f)
        print("✅ Profile saved!")

    def load_profile(self):
        try:
            if os.path.exists("profile_eval.json"):
                with open("profile_eval.json") as f:
                    profile = json.load(f)
                self.dataset_input.setText(profile.get('dataset', ''))
                self.weights_input.setText(profile.get('weights', ''))
                self.output_input.setText(profile.get('output', ''))
                self.limit_field.setText(str(profile.get('limit', '100')))
                self.cpu_checkbox.setChecked(profile.get('use_cpu', False))
                print("✅ Profile loaded!")
            else:
                print("ℹ️ No profile found.")
        except Exception as e:
            print(f"❌ Failed to load profile: {e}")

    def append_log(self, text):
        cursor = self.result_text.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.result_text.setTextCursor(cursor)
        self.result_text.ensureCursorVisible()

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def update_status(self, msg):
        self.status_label.setText(msg)

    def start_evaluation(self):
        dataset_path = self.dataset_input.text()
        weights_path = self.weights_input.text()
        output_path = self.output_input.text()

        if not os.path.exists(dataset_path) or not os.path.exists(weights_path):
            QMessageBox.critical(self, "Error", "Invalid dataset or weights path!")
            return
        
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except:
                pass

        try:
            limit = int(self.limit_field.text())
        except ValueError:
            limit = 100

        self.eval_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.result_text.clear()
        self.status_label.setText("Initializing...")

        self.thread = EvaluationThread(
            dataset_path, 
            weights_path, 
            output_path, 
            limit, 
            self.cpu_checkbox.isChecked()
        )
        
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.status_signal.connect(self.update_status)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.error_signal.connect(self.on_error)
        
        self.thread.start()

    def on_finished(self, result_msg):
        self.eval_btn.setEnabled(True)
        self.status_label.setText("Evaluation Finished")
        QMessageBox.information(self, "Done", "Evaluation Finished successfully.")

    def on_error(self, error_msg):
        self.eval_btn.setEnabled(True)
        self.status_label.setText("Error")
        print(f"❌ CRITICAL ERROR: {error_msg}")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")

    def closeEvent(self, event):
        # Restore original streams on exit
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = EvaluationApp()
    gui.show()
    sys.exit(app.exec_())
