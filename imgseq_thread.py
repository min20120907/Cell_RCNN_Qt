class imgSeqThread(QtCore.QThread):
    def __init__(self, parent=None, WORK_DIR = '',txt='', weight_path = '',dataset_path='',ROI_PATH='',DETECT_PATH='',DEVICE=':/gpu', conf_rate=0.9, epoches=10, step=100):
        super(imgSeqThread, self).__init__(parent)
        self.DETECT_PATH=DETECT_PATH
        self.WORK_DIR = WORK_DIR
        self.weight_path = weight_path
        self.dataset_path = dataset_path
        self.ROI_PATH=ROI_PATH
        self.txt = txt
        self.DEVICE=DEVICE
        self.conf_rate=conf_rate
        self.epoches=epoches
        self.step = step
    append = QtCore.pyqtSignal(str)
    progressBar = QtCore.pyqtSignal(int)
    progressBar_setMaximum = QtCore.pyqtSignal(int)
    def run(self):