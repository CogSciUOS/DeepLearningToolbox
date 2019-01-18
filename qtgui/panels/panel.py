from PyQt5.QtWidgets import QWidget

from observer import Observer
from controller import BaseController

class Panel(QWidget):
    '''Base class for different visualisation panels.
    '''

    def __init__(self, parent: QWidget=None):
        '''Initialization of the ActivationsView.

        Parameters
        ----------
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)

    def setController(self, controller: BaseController,
                      observerType: type=Observer):
        for child in self.findChildren(observerType):
            child.setController(controller)
