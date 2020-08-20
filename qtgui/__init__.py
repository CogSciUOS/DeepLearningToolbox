from toolbox import Toolbox

from . import panels
from . import widgets

from PyQt5.QtWidgets import QApplication


def create_gui(argv=[], toolbox: Toolbox = None, **kwargs):
    """Create the Qt based graphical user interface (GUI) for the
    Toolbox.

    This will create a :py:class:`QApplication` and a
    :py:class:`QMainWindow` (of type :py:class:`DeepVisMainWindow`)
    and show this main window.

    This function will, however, not start the main event loop. This
    can be done by invoking `gui.run()`. That function will only
    return once the main event loop has finished.
    """
    # postpone import of mainwindow for performance reasons
    from .mainwindow import DeepVisMainWindow
    if toolbox.option('firefox_bug'):
        mainwindow.BUG = True

    application = QApplication(argv)
    mainWindow = DeepVisMainWindow(application, toolbox, **kwargs)

    return mainWindow
