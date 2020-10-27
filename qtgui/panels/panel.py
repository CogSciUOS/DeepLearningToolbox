import re

from dltb.base import Observable
from base import Controller as BaseController

from PyQt5.QtWidgets import QWidget, QTabWidget, QStackedWidget


class Panel(QWidget):
    '''Base class for different visualisation panels.
    '''

    def __init__(self, parent: QWidget = None):
        '''Initialization of the ActivationsView.

        Parameters
        ----------
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)

    def setController(self, controller: BaseController,
                      observerType: type = Observable.Observer):
        for child in self.findChildren(observerType):
            child.setController(controller)

    def attention(self, alert=True):
        """Indicate that this panel may need some attention.

        FIXME[hack]: this is just a first quick-and-dirty implementation.
        Do something nicer once there is time ...
        """
        parent = self.parentWidget()
        if parent is None:
            return

        if parent is not None and isinstance(parent, QStackedWidget):
            parent = parent.parentWidget()

        if parent is not None and isinstance(parent, QTabWidget):
            me = parent.indexOf(self)
            text = parent.tabText(me)
            text = re.sub(r'^X ', '', text)
            if alert:
                text = 'X ' + text
            text = parent.setTabText(me, text)
            # parent.item().setTextColor(QColor(color));
