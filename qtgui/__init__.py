from . import panels
from . import widgets

from PyQt5.QtWidgets import QApplication

def create_gui(argv, toolbox_controller):
    from .mainwindow import DeepVisMainWindow
    application = QApplication(argv)
    mainWindow = DeepVisMainWindow(application, toolbox_controller)
    mainWindow.show()

    return mainWindow
