from . import panels
from . import widgets

from PyQt5.QtWidgets import QApplication

def create_gui(argv, toolbox_controller, **kwargs):
    # postpone import of mainwindow for performance reasons
    from .mainwindow import DeepVisMainWindow

    application = QApplication(argv)
    mainWindow = DeepVisMainWindow(application, toolbox_controller, **kwargs)
    mainWindow.show()

    return mainWindow
