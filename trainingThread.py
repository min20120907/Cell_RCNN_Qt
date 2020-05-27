from PyQt5 import QtCore
import time

class trainingThread(QtCore.QThread):
    def __init__(self, parent=None):
        super(trainingThread, self).__init__(parent)
    
    update_training_status = QtCore.pyqtSignal(str)

    def run(self):
        while(True):
            time.sleep(2)
            self.update_training_status.emit('training')