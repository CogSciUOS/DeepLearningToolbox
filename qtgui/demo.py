"""The Qt demo module is intended to test individual widgets from the
Qt-GUI of the Deep Learning ToolBox.

import qtgui.debug
from qtgui.demo import QDemo
demo = QDemo()
demo.show()
"""

# pylint: disable=wrong-import-position
if __name__ == '__main__':
    print(f"Chaning package from '{__package__}' to 'qtqui'")
    __package__ = 'qtgui'  # make relative imports work

# standard imports
import logging

# third-party imports
import numpy as np

# Qt imports
from PyQt5.QtCore import pyqtSignal, pyqtBoundSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QLabel, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout

# toolbox imports
from toolbox import Toolbox
from dltb.network import Network
from dltb.datasource import Datasource
from dltb import thirdparty

# GUI imports
from . import debug
from .utils import QObserver, protect
from .adapter import QAdaptedListWidget, QAdaptedComboBox
from .widgets.register import QRegisterClassView
from .widgets.register import QClassRegisterEntryController
from .widgets.register import QInstanceRegisterEntryController
from .widgets.image import QImageView
from .widgets.network import QLayerSelector
from .widgets.datasource import QDatasourceListWidget, QDatasourceComboBox
from .widgets.features import QFeatureView

# logging
LOG = logging.getLogger(__name__)


class QDemo(QWidget):
    """The :py:class:`QDemo` widget allow to select different widgets
    for demonstration.
    """

    _widgetClasses = [
        QAdaptedListWidget,
        QAdaptedComboBox,
        QRegisterClassView,
        QClassRegisterEntryController,
        QInstanceRegisterEntryController,
        QImageView,
        QLayerSelector,
        QDatasourceListWidget,
        QDatasourceComboBox,
        QFeatureView
    ]

    def __init__(self, **kwargs) -> None:
        """Initialize the :py:class:`QDemo`.
        """
        super().__init__(**kwargs)
        self._widget = None
        self._initUI()
        self._layoutUI()
        self.toolbox = Toolbox()

    def _initUI(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        """Initialize the user interface components of this
        :py:class:`QDemo`.
        """
        self._dummyWidget = QLabel("Select a Widget!")
        self._widget = self._dummyWidget

        self._widgetClassSelector = QAdaptedComboBox()
        self._widgetClassSelector.setItemToText(lambda cls: cls.__name__)
        self._widgetClassSelector.updateFromIterable(self._widgetClasses)
        self._widgetClassSelector.setCurrentIndex(-1)  # deselect entry
        self._widgetClassSelector.\
            currentTextChanged.connect(self.onWidgetClassSelected)

        self._toolboxCheckbox = QCheckBox("Toolbox")
        self._toolboxCheckbox.setEnabled(False)
        self._toolboxCheckbox.stateChanged.connect(self.onToolboxChecked)

    def _layoutUI(self) -> None:
        """Layout the user interface components of this
        :py:class:`QDemo`.
        """
        widgetRow = QHBoxLayout()
        widgetRow.addStretch()
        widgetRow.addWidget(self._widget)
        widgetRow.addStretch()

        layout = QVBoxLayout()
        layout.addWidget(self._widgetClassSelector)
        layout.addStretch()
        layout.addLayout(widgetRow)
        layout.addStretch()
        layout.addWidget(self._toolboxCheckbox)
        self.setLayout(layout)

    def widget(self) -> QWidget:
        """Obtain the currently shown widget. May be `None` if
        currently no widget is selected.
        """
        return None if self._widget is self._dummyWidget else self._widget

    def setWidget(self, widget: QWidget) -> None:
        """Set a widget to be shown as demo.
        """

        if widget is None:
            widget = self._dummyWidget
        if widget is self._widget:
            return  # nothing to do

        # disconnect the old widget
        if self._widget is not self._dummyWidget:
            for name in dir(type(self._widget)):
                attribute = getattr(type(self._widget), name)
                if isinstance(attribute, pyqtSignal):
                    boundSignal: pyqtBoundSignal = getattr(widget, name)
                    boundSignal.disconnect(self.signalHandler)

        layoutItem = self.layout().replaceWidget(self._widget, widget)
        del layoutItem
        self._widget.setParent(None)
        if self._widget is not self._dummyWidget:
            self._widget.deleteLater()
        self._widget = widget

        # self._widgetLayout.removeWidget(self.widget_name)
        # self.widget_name.deleteLater()
        # self.widget_name = None

        widgetClass = type(widget)
        hasToolbox: bool = False
        if isinstance(widget, QObserver):
            print(f"{widgetClass} is a QObserver:")
            print(f"  {len(widget._qobservables)} Observables:")
            for cls, interests in widget._qobservables.items():
                print(f"    {cls}: {interests}")
                if cls is Toolbox:
                    hasToolbox = True
            print(f"  {len(widget._qObserverHelpers)} Observations:")
            for cls, helper in widget._qObserverHelpers.items():
                print(f"    {cls}: {helper}")

        print(f"{widgetClass.__name__} has the following signals:")
        if widget is not self._dummyWidget:
            for name in dir(widgetClass):
                attribute = getattr(widgetClass, name)
                if isinstance(attribute, pyqtSignal):
                    boundSignal: pyqtBoundSignal = getattr(widget, name)
                    boundSignal.connect(self.signalHandler)
                    print(f"  - connected '{name} [{boundSignal}]'")

        # Provide the toolbox checkbox
        self._toolboxCheckbox.setEnabled(hasToolbox)

    # @pyqtSlot
    def signalHandler(self, *args, **kwargs) -> None:
        """
        """
        print(f"signalHandler(args={args}, kwargs={kwargs})")

    @protect
    def onWidgetClassSelected(self, widgetClassName: str) -> None:
        """Slot to react to a change of the selected widget in their
        widget class combo box.
        """

        demoMethod = getattr(self, 'demo' + widgetClassName, None)
        if demoMethod is not None:
            demoMethod()
        else:
            widgetClass = next((cls for cls in self._widgetClasses
                                if cls.__name__ == widgetClassName), None)
            if widgetClass is not None:
                self.setWidget(widgetClass())
            else:
                self.setWidget(None)

    @protect
    def onToolboxChecked(self, state: int) -> None:
        """Slot to set or unset the Toolbox for a widget.
        """
        self._widget.setToolbox(self.toolbox if bool(state) else None)

    #
    # Demo methods
    #

    def demoQAdaptedListWidget(self) -> None:
        """Demonstration of the :py:class:`QAdaptedListWidget`.
        """
        widget = QAdaptedListWidget()
        widget.setItemToText(lambda item: item.__name__)
        widget.setFromIterable(self._widgetClasses)
        self.setWidget(widget)

    def demoQAdaptedComboBox(self) -> None:
        """Demonstration of the :py:class:`QAdaptedComboBox`.
        """
        widget = QAdaptedComboBox()
        widget.setItemToText(lambda item: item.__name__)
        widget.setFromIterable(self._widgetClasses)
        self.setWidget(widget)

    def demoQRegisterClassView(self) -> None:
        """Demonstration of the :py:class:`QRegisterClassView`.
        """
        widget = QRegisterClassView()
        widget.setRegisterClass(Datasource)
        self.setWidget(widget)

    def demoQClassRegisterEntryController(self) -> None:
        """Demonstration of the :py:class:`QClassRegisterEntryController`.
        """
        widget = QClassRegisterEntryController(Datasource.class_register)
        # setting a entry should change the state of the widget
        widget.setRegisterEntry('datasource.widerface.WiderFace')

        # importing the class should be noticed by the widget.
        # from dltb.thirdparty.datasource.widerface import WiderFace
        self.setWidget(widget)

    def demoQInstanceRegisterEntryController(self) -> None:
        """Demonstration of the :py:class:`QInstanceRegisterEntryController`.
        """
        widget = QInstanceRegisterEntryController(Datasource.instance_register)

        # setting a entry should change the state of the widget
        widget.setRegisterEntry('widerface')

        # importing the class should be noticed by the widget.
        # from dltb.thirdparty.datasource.widerface import WiderFace
        self.setWidget(widget)

    def demoQImageView(self) -> None:
        """Demonstration of the :py:class:`QImageView`.
        """
        widget = QImageView()
        thirdparty.import_class('ImageReader', module='imageio')
        widget.setImagelike('https://upload.wikimedia.org/wikipedia/commons/'
                            'thumb/b/b4/Last_Supper_by_Leonardo_da_Vinci.jpg/'
                            '320px-Last_Supper_by_Leonardo_da_Vinci.jpg')
        self.setWidget(widget)

    def demoQLayerSelector(self) -> None:
        """Demonstration of the :py:class:`QLayerSelector`.
        """
        widget = QLayerSelector()
        network = Network['alexnet-tf']
        widget.setNetwork(network)
        self.setWidget(widget)

    def demoQDatasourceListWidget(self) -> None:
        """Demonstration of the :py:class:`QDatasourceListWidget`.
        """
        widget = QDatasourceListWidget()
        print(type(widget.datasourceSelected))
        print(type(type(widget).datasourceSelected))
        print(type(widget.datasourceSelected).__mro__)
        self.setWidget(widget)

    def demoQDatasourceComboBox(self) -> None:
        """Demonstration of the :py:class:`QDatasourceComboBox`.
        """
        widget = QDatasourceComboBox()
        self.setWidget(widget)

    def demoQFeatureView(self) -> None:
        """Demonstration of the :py:class:`QDatasourceComboBox`.
        """
        widget = QFeatureView()
        widget.setFeatures(np.random.randn(512))
        self.setWidget(widget)


if __name__ == '__main__':
    demo = QDemo()
    demo.show()
    result = debug.application.exec_()
