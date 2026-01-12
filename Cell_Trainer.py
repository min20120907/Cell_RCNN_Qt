import sys
import json
import ray
from os.path import dirname
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

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

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_title("Training Loss")
        self.axes.set_xlabel("Steps")
        self.axes.set_ylabel("Loss")
        super(MplCanvas, self).__init__(self.fig)

# --- 1. Robust Stream Redirector (Fixes Ray fileno & Cleans Output) ---
class StreamRedirector(QtCore.QObject):
    """
    Wraps sys.stdout/stderr. 
    1. Implements fileno() to satisfy Ray/Faulthandler.
    2. Filters text to prevent GUI spam (e.g., skips carriage returns).
    """
    message = QtCore.pyqtSignal(str)

    def __init__(self, original_stream, parent=None):
        super(StreamRedirector, self).__init__(parent)
        self.original_stream = original_stream

    def write(self, text):
        # Filter out common progress bar characters that cause spam in text widgets
        if not text: return
        
        # If text is just a carriage return or empty newline, we might want to skip logic depending on preference
        # For a clean log, we send it to GUI only if it has content
        if text.strip() or "\n" in text:
             self.message.emit(str(text))

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except: pass

    def fileno(self):
        """Required by Ray and faulthandler."""
        if self.original_stream:
            return self.original_stream.fileno()
        return 1 

    def isatty(self):
        """Required by some logging libraries."""
        if self.original_stream:
            return self.original_stream.isatty()
        return False

    def __getattr__(self, name):
        """Delegate other methods to the original stream."""
        return getattr(self.original_stream, name)


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
        # --- ADD THIS: Setup Graph and Status Bar ---

        # 1. Create Layout for Graph if you don't have a place for it
        # Assuming you have a central widget or a layout to add to.
        # If you defined a placeholder in Qt Designer, use it.
        # Here I create a new Layout and add it to a specific area (e.g. textBrowser's parent)

        # Example: finding a layout to add the graph (adjust 'verticalLayout' to your actual layout name)
        # self.graph_layout = self.findChild(QVBoxLayout, 'verticalLayout') 

        # Or creating a dedicated widget area:
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.loss_data = [] # Store loss history

        # Add to your existing layout (replace 'self.verticalLayout' with your actual layout)
        # If you don't know the layout, you can add it below your textBrowser usually
        if hasattr(self, 'verticalLayout'): 
            self.verticalLayout.addWidget(self.canvas)
        else:
            # Fallback: try to add to central widget layout
            self.centralWidget().layout().addWidget(self.canvas)

        # 2. Setup Status Bar Label
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label) # Add to bottom status bar

        # --- END ADDITION ---

        # --- 2. Setup Output Redirection ---
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

    # --- 3. Slot to Handle Terminal Text ---
    def on_terminal_output(self, text):
        """Efficiently appends text to the GUI."""
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

    # --- Profile Handling ---
    def load_profile(self):
        try:
            with open("profile.json") as f:
                data = json.loads(f.read())
            f = data
            self.epoches = f['epoches']
            self.epochs.setText(str(f['epoches']))
            self.confidence = f['confidence']
            self.conf_rate.setText(str(f['confidence']))
            self.DEVICE = f['DEVICE']
            if(self.DEVICE == "/cpu:0"):
                self.cpu_train.toggle()
            elif(self.DEVICE == "/gpu:0"):
                self.gpu_train.toggle()
            self.dataset_path = f['dataset_path']
            self.WORK_DIR = f['WORK_DIR']
            self.ROI_PATH = f['ROI_PATH']
            self.DETECT_PATH = f['DETECT_PATH']
            self.coco_path = f['coco_path']
            self.weight_path = f['weight_path']
            self.steps_num = f['steps']
            self.format_txt.setText(f['txt'])
            self.steps.setText(str(f['steps']))
            self.is_path = f['is_path']
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
    def update_status_bar(self, msg):
        """Slot to update the status bar text"""
        self.status_label.setText(msg)

    def update_loss_graph(self, loss):
        """Slot to update the loss graph"""
        self.loss_data.append(loss)
        self.canvas.axes.cla()  # Clear previous plot
        self.canvas.axes.plot(self.loss_data, 'r-', label='Loss')
        self.canvas.axes.set_title("Training Loss")
        self.canvas.axes.legend()
        self.canvas.draw()
    # --- Worker Thread Helpers ---
    def train_t(self):
        try:
            self.epoches = int(self.epochs.text())
            self.confidence = float(self.conf_rate.text())
            
            self.myThread = QtCore.QThread()
            self.thread = trainingThread.trainingThread(
                test=1, 
                steps=self.steps_num, 
                train_mode=self.train_mode.text(), 
                dataset_path=self.dataset_path, 
                confidence=self.confidence, 
                epoches=self.epoches, 
                WORK_DIR=self.WORK_DIR, 
                weight_path=self.weight_path
            )
            self.thread.update_training_status.connect(self.append)
            # Connect GUI Progress Bar
            self.thread.progressBar.connect(self.progressBar.setValue)
            self.thread.progressBar_setMaximum.connect(self.progressBar.setMaximum)
            
            # --- CONNECT NEW SIGNALS ---
            self.thread.update_status_bar.connect(self.update_status_bar)
            self.thread.update_plot_data.connect(self.update_loss_graph)
            # ---------------------------

            self.thread.moveToThread(self.myThread)
            self.myThread.started.connect(self.thread.run)
            self.myThread.start()
        except Exception as e:
            self.append(f"Error starting training: {e}")

    def detect(self):
        try:
            self.myThread = QtCore.QThread()
            self.thread = detectingThread.detectingThread(
                DETECT_PATH=self.DETECT_PATH, 
                ROI_PATH=self.ROI_PATH, 
                txt=self.format_txt.text(), 
                weight_path=self.weight_path, 
                dataset_path=self.dataset_path, 
                WORK_DIR=self.WORK_DIR,
                DEVICE=self.DEVICE,
                conf_rate=self.conf_rate.text(),
                epoches=self.epochs.text(),
                step=self.steps.text()
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
            DETECT_PATH=self.DETECT_PATH, 
            ROI_PATH=self.ROI_PATH, 
            txt=self.format_txt.text(), 
            weight_path=self.weight_path, 
            dataset_path=self.dataset_path, 
            WORK_DIR=self.WORK_DIR,
            DEVICE=self.DEVICE,
            conf_rate=self.conf_rate.text(),
            epoches=self.epochs.text(),
            step=self.steps.text() 
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Cell()
    window.show()
    sys.exit(app.exec_())
