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


class QStandalone:
    """A class to run a single Qt widget standalone.  This requires
    a `QApplication` object to be initialized, which can then be
    used to run the main event loop.


    Class properties
    ================
    `qapplication`:
        The global `QApplication` object that can be used to run the main
        event loop.
    """
    qapplication: QApplication = None

    @staticmethod
    def ensureQApplication() -> None:
        """Ensure that the `QApplication` object required for running
        an event loop, was initialized.
        """
        if QStandalone.qapplication is None:
            QStandalone.qapplication = QApplication([])

    def __new__(cls, **kwargs) -> 'QStandalone':
        cls.ensureQApplication()
        __new__ = super().__new__
        return  (__new__(cls) if __new__ is object.__new__ else
                 __new__(cls, **kwargs))

    def showStandalone(self) -> None:
        """Show the Widget and run the main event loop.
        """
        self.qapplication.exec_()
