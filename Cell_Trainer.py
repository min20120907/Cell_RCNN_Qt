import sys
import json
import ray
import os
import subprocess # 用來啟動 tensorboard
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QApplication, QFileDialog, QLabel, 
                             QVBoxLayout, QWidget, QTabWidget, QScrollArea, 
                             QGridLayout, QFrame, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QUrl

# 🔥 [新增] WebEngine 用於顯示 TensorBoard
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
except ImportError:
    print("❌ 錯誤: 找不到 PyQtWebEngine。請執行: pip install PyQtWebEngine")
    # 為了防止程式崩潰，定義一個假的 QWebEngineView (如果沒安裝的話)
    class QWebEngineView(QLabel):
        def setUrl(self, url): self.setText(f"請安裝 PyQtWebEngine 以檢視: {url}")
        def load(self, url): self.setText(f"請安裝 PyQtWebEngine 以檢視: {url}")

# Matplotlib integration
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Import Threads
import batchDetectThread
import batch_cocoThread
import batch_cocoShrinkThread
import anotThread
import BWThread
import cocoThread
import detectingThread
import trainingThread
import imgseq_thread
from main_ui import Ui_MainWindow

# --- 1. Loss Graph Canvas ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_title("Training Loss")
        self.axes.set_xlabel("Steps")
        self.axes.set_ylabel("Loss")
        self.axes.grid(True)
        super(MplCanvas, self).__init__(self.fig)

# --- 2. Robust Stream Redirector ---
class StreamRedirector(QtCore.QObject):
    message = QtCore.pyqtSignal(str)
    def __init__(self, original_stream, parent=None):
        super(StreamRedirector, self).__init__(parent)
        self.original_stream = original_stream
    def write(self, text):
        if not text: return
        if text.strip() or "\n" in text: self.message.emit(str(text))
    def flush(self):
        if self.original_stream:
            try: self.original_stream.flush()
            except: pass
    def fileno(self):
        if self.original_stream: return self.original_stream.fileno()
        return 1 
    def isatty(self):
        return False
    def __getattr__(self, name):
        return getattr(self.original_stream, name)

# --- 3. Main Window ---
class Cell(QMainWindow, Ui_MainWindow):
    epoches = 100
    confidence = 0.9
    DEVICE = "/cpu:0"
    dataset_path = ""
    weight_path = ""
    WORK_DIR = ""
    ROI_PATH = ""
    DETECT_PATH = ""
    coco_path = ""
    steps_num = 1
    is_path = ""

    def __init__(self, parent=None):
        super(Cell, self).__init__(parent)
        self.setupUi(self)
        
        # TensorBoard Process 變數
        self.tb_process = None
        
        # ==========================================================
        # 🔥 UI 改造區：新增 TensorBoard 分頁 🔥
        # ==========================================================
        
        # 1. 建立 Tab Widget
        self.viz_tabs = QTabWidget()
        self.viz_tabs.setMinimumHeight(500) 

        # --- Tab 1: Loss Graph ---
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.loss_data = {'loss': [], 'rpn_bbox': [], 'mrcnn_bbox': []} 
        self.viz_tabs.addTab(self.canvas, "📈 Training Loss")

        # --- Tab 2: Gallery ---
        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout() 
        self.gallery_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.gallery_widget.setLayout(self.gallery_layout)
        self.gallery_scroll.setWidget(self.gallery_widget)
        self.viz_tabs.addTab(self.gallery_scroll, "🖼️ Live Gallery")

        # --- Tab 3: TensorBoard (NEW!) ---
        self.tb_webview = QWebEngineView()
        # 預設顯示提示訊息
        self.tb_webview.setHtml("""
        <h2 style='color:white; font-family:sans-serif; text-align:center; margin-top:50px;'>
            TensorBoard is waiting to start...<br>
            Please set Working Directory and click 'Train'.
        </h2>
        """, QUrl(""))
        self.tb_webview.setStyleSheet("background-color: #2b2b2b;")
        self.viz_tabs.addTab(self.tb_webview, "📊 TensorBoard")

        # 將 Tab 加入介面
        if hasattr(self, 'verticalLayout'): 
            self.verticalLayout.addWidget(self.viz_tabs)
        elif hasattr(self, 'centralwidget') and self.centralwidget.layout():
            self.centralwidget.layout().addWidget(self.viz_tabs)
        else:
            self.textBrowser.parentWidget().layout().addWidget(self.viz_tabs)

        # 2. Status Bar Label
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

        # ==========================================================
        # Stream Redirection
        # ==========================================================
        self.stdout_original = sys.stdout
        self.stderr_original = sys.stderr
        self.redirector_stdout = StreamRedirector(self.stdout_original)
        self.redirector_stderr = StreamRedirector(self.stderr_original)
        self.redirector_stdout.message.connect(self.on_terminal_output)
        self.redirector_stderr.message.connect(self.on_terminal_output)
        sys.stdout = self.redirector_stdout
        sys.stderr = self.redirector_stderr

        # Button Events
        self.train_btn.clicked.connect(self.train_t)
        self.detect_btn.clicked.connect(self.detect)
        self.batch_snc.clicked.connect(self.sncBatch)
        self.gpu_train.clicked.connect(self.gpu_train_func)
        self.cpu_train.clicked.connect(self.cpu_train_func)
        self.clear_logs.clicked.connect(self.clear)
        self.upload_sets.clicked.connect(self.get_sets)
        self.upload_weight.clicked.connect(self.get_weight)
        self.upload_det.clicked.connect(self.get_detect)
        self.mrcnn_btn.clicked.connect(self.get_mrcnn)
        self.output_dir.clicked.connect(self.save_ROIs)
        self.roi_convert.clicked.connect(self.zip2coco)
        self.l_profile.clicked.connect(self.load_profile)
        self.s_profile.clicked.connect(self.save_profile)
        self.batch_coco.clicked.connect(self.cocoBatch)
        self.batch_detect.clicked.connect(self.detectBatch)
        self.is_btn.clicked.connect(self.exec_is_path)

    # --- Slot to Handle Terminal Text ---
    def on_terminal_output(self, text):
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def append(self, a):
        now = datetime.now()
        current_time = now.strftime("[%m-%d-%Y %H:%M:%S] ")
        print(current_time + str(a))

    def clear(self):
        self.textBrowser.clear()
        self.loss_data = {'loss': [], 'rpn_bbox': [], 'mrcnn_bbox': []}
        self.canvas.axes.cla()
        self.canvas.draw()
        for i in reversed(range(self.gallery_layout.count())): 
            self.gallery_layout.itemAt(i).widget().setParent(None)

    # --- Profile Handling ---
    def load_profile(self):
        try:
            if os.path.exists("profile.json"):
                with open("profile.json") as f:
                    f = json.load(f)
                self.epoches = f['epoches']
                self.epochs.setText(str(f['epoches']))
                self.confidence = f['confidence']
                self.conf_rate.setText(str(f['confidence']))
                self.DEVICE = f['DEVICE']
                if(self.DEVICE == "/cpu:0"): self.cpu_train.toggle()
                elif(self.DEVICE == "/gpu:0"): self.gpu_train.toggle()
                self.dataset_path = f['dataset_path']
                self.WORK_DIR = f['WORK_DIR']
                self.ROI_PATH = f['ROI_PATH']
                self.DETECT_PATH = f['DETECT_PATH']
                self.coco_path = f['coco_path']
                self.weight_path = f['weight_path']
                self.steps_num = f['steps']
                self.format_txt.setText(f['txt'])
                self.steps.setText(str(f['steps']))
                self.is_path = f.get('is_path', '')
                print("Json profile loaded!")
        except Exception as e:
            print(f"Error loading profile: {e}")

    def save_profile(self):
        try:
            tmp = dict()
            tmp['epoches'] = int(self.epochs.text())
            tmp['confidence'] = float(self.conf_rate.text())
            tmp['DEVICE'] = self.DEVICE
            tmp['dataset_path'] = self.dataset_path
            tmp['WORK_DIR'] = self.WORK_DIR
            tmp['ROI_PATH'] = self.ROI_PATH
            tmp['DETECT_PATH'] = self.DETECT_PATH
            tmp['coco_path'] = self.coco_path
            tmp['weight_path'] = self.weight_path
            tmp['steps'] = self.steps.text()
            tmp['txt'] = self.format_txt.text()
            tmp['is_path'] = self.is_path
            with open('profile.json', 'w') as json_file:
                json.dump(tmp, json_file)
            print("Json Profile saved!")
        except Exception as e:
            print(f"Error saving profile: {e}")

    # ==========================================================
    # 🔥 TensorBoard 啟動邏輯 🔥
    # ==========================================================
    def launch_tensorboard(self):
        """在背景啟動 TensorBoard 並將 WebEngine 指向它"""
        if self.WORK_DIR == "":
            self.append("⚠️ Cannot start TensorBoard: Working Directory is not set.")
            return

        log_dir = os.path.join(self.WORK_DIR, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 如果已經有開過的，先殺掉重開 (確保路徑正確)
        if self.tb_process:
            self.tb_process.kill()
            self.tb_process = None

        port = "6006" # 預設 TensorBoard Port
        self.append(f"🚀 Starting TensorBoard on port {port} at {log_dir}...")
        
        try:
            # 啟動 subprocess
            self.tb_process = subprocess.Popen(
                ["tensorboard", "--logdir", log_dir, "--port", port, "--reload_interval", "30"],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            # 載入網頁 (稍等一下讓 server 啟動)
            QtCore.QTimer.singleShot(2000, lambda: self.tb_webview.setUrl(QUrl(f"http://localhost:{port}")))
            
        except Exception as e:
            self.append(f"❌ Failed to start TensorBoard: {e}")

    # ==========================================================
    # UI 更新 Slots
    # ==========================================================
    def update_status_bar(self, msg):
        self.status_label.setText(msg)

    def update_loss_graph(self, loss_obj):
        self.canvas.axes.cla()
        if isinstance(loss_obj, dict):
            total_loss = loss_obj.get('loss', 0)
            self.loss_data['loss'].append(total_loss)
            if 'rpn_bbox_loss' in loss_obj:
                self.loss_data['rpn_bbox'].append(loss_obj['rpn_bbox_loss'])
            if 'mrcnn_bbox_loss' in loss_obj:
                self.loss_data['mrcnn_bbox'].append(loss_obj['mrcnn_bbox_loss'])
            self.canvas.axes.plot(self.loss_data['loss'], 'r-', label='Total Loss', linewidth=2)
            if self.loss_data['rpn_bbox']:
                self.canvas.axes.plot(self.loss_data['rpn_bbox'], 'g--', label='RPN BBox', alpha=0.6)
            if self.loss_data['mrcnn_bbox']:
                self.canvas.axes.plot(self.loss_data['mrcnn_bbox'], 'b--', label='MRCNN BBox', alpha=0.6)
        else:
            self.loss_data['loss'].append(loss_obj)
            self.canvas.axes.plot(self.loss_data['loss'], 'r-', label='Loss')
        self.canvas.axes.set_title("Training Loss Metrics")
        self.canvas.axes.set_xlabel("Steps")
        self.canvas.axes.set_ylabel("Loss")
        self.canvas.axes.legend(loc='upper right')
        self.canvas.axes.grid(True, linestyle='--', alpha=0.5)
        self.canvas.draw()

    def update_gallery(self, image_path, index):
        pixmap = QPixmap(image_path)
        if pixmap.isNull(): return
        scaled_pixmap = pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_label = QLabel()
        img_label.setPixmap(scaled_pixmap)
        img_label.setFrameShape(QFrame.StyledPanel)
        txt_label = QLabel(f"Sample {index}")
        txt_label.setAlignment(Qt.AlignCenter)
        cell_widget = QWidget()
        cell_layout = QVBoxLayout()
        cell_layout.addWidget(img_label)
        cell_layout.addWidget(txt_label)
        cell_widget.setLayout(cell_layout)
        item = self.gallery_layout.itemAtPosition(0, index)
        if item:
            widget_to_remove = item.widget()
            widget_to_remove.setParent(None)
        self.gallery_layout.addWidget(cell_widget, 0, index)

    # --- Worker Thread Helpers ---
    def train_t(self):
        try:
            self.epoches = int(self.epochs.text())
            self.confidence = float(self.conf_rate.text())
            try: self.steps_num = int(self.steps.text())
            except: self.steps_num = 1000

            # 🔥 [NEW] 啟動 TensorBoard
            self.launch_tensorboard()

            self.myThread = QtCore.QThread()
            self.thread = trainingThread.trainingThread(
                test=1, steps=self.steps_num, train_mode=self.train_mode.text(), 
                dataset_path=self.dataset_path, confidence=self.confidence, 
                epoches=self.epoches, WORK_DIR=self.WORK_DIR, weight_path=self.weight_path
            )
            
            self.thread.update_training_status.connect(self.append)
            self.thread.progressBar.connect(self.progressBar.setValue)
            self.thread.progressBar_setMaximum.connect(self.progressBar.setMaximum)
            self.thread.update_status_bar.connect(self.update_status_bar)
            self.thread.update_plot_data.connect(self.update_loss_graph)
            self.thread.update_gallery_signal.connect(self.update_gallery)

            self.thread.moveToThread(self.myThread)
            self.myThread.started.connect(self.thread.run)
            self.myThread.start()
            
            # 切換到 TensorBoard 或 Graph
            self.viz_tabs.setCurrentIndex(2) # 0:Graph, 1:Gallery, 2:TensorBoard
            
        except Exception as e:
            self.append(f"Error starting training: {e}")

    def detect(self):
        try:
            self.myThread = QtCore.QThread()
            self.thread = detectingThread.detectingThread(
                DETECT_PATH=self.DETECT_PATH, ROI_PATH=self.ROI_PATH, 
                txt=self.format_txt.text(), weight_path=self.weight_path, 
                dataset_path=self.dataset_path, WORK_DIR=self.WORK_DIR,
                DEVICE=self.DEVICE, conf_rate=self.conf_rate.text(),
                epoches=self.epochs.text(), step=self.steps.text()
            )
            self.thread.append.connect(self.append)
            self.thread.progressBar.connect(self.progressBar.setValue)
            self.thread.progressBar_setMaximum.connect(self.progressBar.setMaximum)
            self.thread.moveToThread(self.myThread)
            self.myThread.started.connect(self.thread.run)
            self.myThread.start()
        except Exception as e:
            self.append(f"Error starting detection: {e}")

    def detectBatch(self):
        self.myThread = QtCore.QThread()
        self.thread = batchDetectThread.batchDetectThread(
            DETECT_PATH=self.DETECT_PATH, ROI_PATH=self.ROI_PATH, 
            txt=self.format_txt.text(), weight_path=self.weight_path, 
            dataset_path=self.dataset_path, WORK_DIR=self.WORK_DIR,
            DEVICE=self.DEVICE, conf_rate=self.conf_rate.text(),
            epoches=self.epochs.text(), step=self.steps.text() 
        )
        self.thread.append.connect(self.append)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.thread.progressBar.connect(self.progressBar.setValue)
        self.thread.progressBar_setMaximum.connect(self.progressBar.setMaximum)
        self.myThread.start()

    def gpu_train_func(self):
        self.append("Training in GPU...")
        self.DEVICE = "/gpu:0"

    def cpu_train_func(self):
        self.append("Training in CPU...")
        self.DEVICE = "/cpu:0"

    def get_sets(self):
        dir_choose = QFileDialog.getExistingDirectory(self, "Select an input directory...", self.dataset_path)
        if dir_choose == "": return
        self.append(f"Selected: {dir_choose}")
        self.dataset_path = dir_choose
    
    def get_output(self):
        dir_choose = QFileDialog.getExistingDirectory(self, "Select an output directory...", self.output_path)
        if dir_choose == "": return
        self.append(f"Selected: {dir_choose}")
        self.output_dir = dir_choose

    def get_detect(self):
        dir_choose = QFileDialog.getExistingDirectory(self, "Select an detecting directory...", self.DETECT_PATH)
        if dir_choose == "": return
        self.append(f"Selected: {dir_choose}")
        self.DETECT_PATH = dir_choose

    def get_is_path(self):
        dir_choose = QFileDialog.getExistingDirectory(self, "Select an Image Sequence directory...", self.DETECT_PATH)
        if dir_choose == "": return
        self.append(f"Selected: {dir_choose}")
        self.is_path = dir_choose

    def get_mrcnn(self):
        dir_choose = QFileDialog.getExistingDirectory(self, "Select an working directory...", self.WORK_DIR)
        if dir_choose == "": return
        self.append(f"Selected: {dir_choose}")
        self.WORK_DIR = dir_choose

    def get_coco(self):
        dir_choose = QFileDialog.getExistingDirectory(self, "Select an COCO directory...", self.coco_path)
        if dir_choose == "": return
        self.append(f"Selected: {dir_choose}")
        self.coco_path = dir_choose

    def get_weight(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self, "Select Weight...", self.weight_path, " COCO Weight Files (*.h5)")
        if fileName_choose == "": return
        self.append(f"Selected Weight: {fileName_choose}")
        self.weight_path = fileName_choose

    def save_ROIs(self):
        dir_choose = QFileDialog.getExistingDirectory(self, "Select an COCO directory...", self.ROI_PATH)
        if dir_choose == "": return
        self.append(f"Selected: {dir_choose}")
        self.ROI_PATH = dir_choose

    def zip2coco(self):
        self.get_coco()
        self.myThread = QtCore.QThread()
        self.thread = cocoThread.cocoThread(coco_path=self.coco_path, txt=self.format_txt.text())
        self.thread.append_coco.connect(self.append)
        self.thread.progressBar.connect(self.progressBar.setValue)
        self.thread.progressBar_setMaximum.connect(self.progressBar.setMaximum)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.myThread.start()

    def exec_is_path(self):
        self.get_is_path()
        self.myThread = QtCore.QThread()
        self.thread = imgseq_thread.imgseq_thread(is_path=self.is_path, txt=self.format_txt.text())
        self.thread.append.connect(self.append)
        self.thread.progressBar.connect(self.progressBar.setValue)
        self.thread.progressBar_setMaximum.connect(self.progressBar.setMaximum)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.myThread.start()

    def cocoBatch(self):
        self.get_coco()
        self.myThread = QtCore.QThread()
        self.thread = batch_cocoThread.batch_cocoThread(coco_path=self.coco_path, txt=self.format_txt.text())
        self.thread.append_coco.connect(self.append)
        self.thread.progressBar.connect(self.progressBar.setValue)
        self.thread.progressBar_setMaximum.connect(self.progressBar.setMaximum)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.myThread.start()

    def sncBatch(self):
        self.get_coco()
        self.myThread = QtCore.QThread()
        self.thread = batch_cocoShrinkThread.batch_sncThread(coco_path=self.coco_path, txt=self.format_txt.text())
        self.thread.append_coco.connect(self.append)
        self.thread.progressBar.connect(self.progressBar.setValue)
        self.thread.progressBar_setMaximum.connect(self.progressBar.setMaximum)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.myThread.start()

    def detect_anot(self):
        self.myThread = QtCore.QThread()
        self.thread = anotThread.anotThread()
        self.thread.append.connect(self.append)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.myThread.start()

    def detect_BW(self):
        self.myThread = QtCore.QThread()
        self.thread = BWThread.BWThread()
        self.thread.append.connect(self.append)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.myThread.start()

    # --- Close Event: Kill TensorBoard ---
    def closeEvent(self, event):
        if self.tb_process:
            self.tb_process.kill()
        sys.stdout = self.stdout_original
        sys.stderr = self.stderr_original
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Cell()
    window.show()
    sys.exit(app.exec_())
