import sys
from PyQt5 import QtWidgets

app = QtWidgets.QApplication(sys.argv)
win = QtWidgets.QWidget()
win.show()
sys.exit(app.exec_())