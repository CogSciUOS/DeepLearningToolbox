"""The Qt demo module is intended to test individual widgets from the
Qt-GUI of the Deep Learning ToolBox.


Run demo (command line)
-----------------------

python -m qtgui.demo


Run demo interactive
--------------------

import qtgui.debug
from qtgui.demo import QDemo
demo = QDemo()
demo.show()


Access the currently active Widget:

widget = demo.widget()

"""

# hack to make relative imports work when called from the command line.
# pylint: disable=wrong-import-position
if __name__ == '__main__':
    print(f"Chaning package from '{__package__}' to 'qtqui'")
    __package__ = 'qtgui'  # pylint: disable=redefined-builtin


# standard imports
import logging

# third-party imports
import numpy as np

# Qt imports
from PyQt5.QtCore import pyqtSignal, pyqtBoundSignal, pyqtSlot
from PyQt5.QtGui import QPaintEvent, QIcon
from PyQt5.QtWidgets import QWidget, QLabel, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout

# toolbox imports
from toolbox import Toolbox
from dltb.base.run import runnable
from dltb.network import Network
from dltb.datasource import Datasource, Noise
from dltb.tool.train import Trainer
from dltb.tool.tests.test_train import MockupTrainee

# GUI imports
from . import debug
from .utils import QObserver, protect, QBusyWidget
from .adapter import QAdaptedListWidget, QAdaptedComboBox
from .widgets.register import QRegisterClassView
from .widgets.register import QClassRegisterEntryController
from .widgets.register import QInstanceRegisterEntryController
from .widgets.image import QImageView
from .widgets.network import QLayerSelector
from .widgets.datasource import QDatasourceListWidget, QDatasourceComboBox
from .widgets.features import QFeatureView
from .widgets.training import QTrainingBox

# logging
LOG = logging.getLogger(__name__)


class QDemo(QWidget):
    """The :py:class:`QDemo` widget allow to select different widgets
    for demonstration.

    Extending the demo
    ------------------
    In order to extend the demo with another class QMyWidget,
    perform the following steps:
    1. import QMyWidget at the haead of the file
    2. add the class QMyWidget to the `_widgetClasses` list below.
    3. implment a new demQMyWidget at the bottom of the class.
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
        QFeatureView,
        QTrainingBox
    ]

    def __init__(self, **kwargs) -> None:
        """Initialize the :py:class:`QDemo`.
        """
        super().__init__(**kwargs)
        self._widget = None
        self.setWindowTitle("QtGUI Demo")
        self.setWindowIcon(QIcon('assets/logo.png'))
        self._initUI()
        self._layoutUI()
        self.toolbox = Toolbox()
        self._debugSignals = True
        self._finishArgs = None
        self._finishMethod = None

    def _initUI(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        """Initialize the user interface components of this
        :py:class:`QDemo`.
        """
        self._dummyWidget = QLabel("Select a Widget!")
        self._busyWidget = QBusyWidget()
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
            self._disconnectAllSignals(self._widget)

        layoutItem = self.layout().replaceWidget(self._widget, widget)
        del layoutItem
        self._widget.setParent(None)
        if self._widget is self._busyWidget:
            self._widget.setBusy(False)
        elif self._widget is not self._dummyWidget:
            self._widget.deleteLater()
        self._widget = widget

        # self._widgetLayout.removeWidget(self.widget_name)
        # self.widget_name.deleteLater()
        # self.widget_name = None

        # widgetClass: type of the new widget
        widgetClass = type(widget)

        # hasToolbox: a flag indicating if the new widget observes a Toolbox
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

        if widget is not self._dummyWidget:
            self._connectAllSignals(widget)

        # Provide the toolbox checkbox
        self._toolboxCheckbox.setEnabled(hasToolbox)

    def _connectAllSignals(self, widget: QWidget, handler=None) -> None:
        """Connect the given handler to all signals of the given
        widget.
        """
        widgetClass = type(widget)
        if handler is None:
            handler = self.signalHandler
        print(f"{widgetClass.__name__} has the following signals:")
        for name in dir(widgetClass):
            attribute = getattr(widgetClass, name)
            if isinstance(attribute, pyqtSignal):
                boundSignal: pyqtBoundSignal = getattr(widget, name)
                boundSignal.connect(handler)
                print(f"  - connected '{name} [{boundSignal}]'"
                      f" - {attribute is boundSignal}")

    def _disconnectAllSignals(self, widget: QWidget, handler=None) -> None:
        """Disconnect the given handler from all signals of the given
        widget.
        """
        widgetClass = type(widget)
        if handler is None:
            handler = self.signalHandler
        for name in dir(widgetClass):
            attribute = getattr(widgetClass, name)
            if isinstance(attribute, pyqtSignal):
                boundSignal: pyqtBoundSignal = getattr(widget, name)
                print(f"  - disconnected '{name} [{boundSignal}]'"
                      f" - {attribute is boundSignal}")
                boundSignal.disconnect(handler)

    # @pyqtSlot()
    # we can not use the decorator @pyqtSlot() here, as that decorator
    # expects the signature of the signal to be known - when used without,
    # arguments, it will just remove all parameters passed to the slot.
    def signalHandler(self, *args, **kwargs) -> None:
        """A general signal handler. It will be connected to
        all signals of a demo class for debugging signals emitted
        by that class.
        """
        if self._debugSignals:
            print(f"signalHandler(args={args}, kwargs={kwargs})")

    @protect
    def onWidgetClassSelected(self, widgetClassName: str) -> None:
        """Slot to react to a change of the selected widget in their
        widget class combo box.
        """

        demoMethod = getattr(self, 'demo' + widgetClassName, None)
        self._finishMethod = \
            getattr(self, 'demoFinish' + widgetClassName, None)
        self._finishArgs = None
        if demoMethod is not None:
            if self._finishMethod is None:
                demoMethod()
            else:
                self._busyWidget.setBusy(True)
                self.setWidget(self._busyWidget)
                demoMethod(run_callback=self._demoFinish)
        else:
            widgetClass = next((cls for cls in self._widgetClasses
                                if cls.__name__ == widgetClassName), None)
            if widgetClass is not None:
                self.setWidget(widgetClass())
            else:
                self.setWidget(None)

    def paintEvent(self, event: QPaintEvent) -> None:
        """React to a paint event by updating the widget.
        """
        if self._finishArgs is not None:
            print("Finishing demo: step 2")
            self._finishMethod(*self._finishArgs)
            self._finishMethod = self._finishArgs = None
        super().paintEvent(event)

    def _demoFinish(self, *args) -> None:
        print("Finishing demo: step 1", type(args))
        self._finishArgs = args
        self.update()

    @protect
    def onToolboxChecked(self, state: int) -> None:
        """Slot to set or unset the Toolbox for a widget.
        """
        self._widget.setToolbox(self.toolbox if bool(state) else None)

    #
    # Demo methods
    #

    @runnable
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
        # We need an imageread that can read URLs
        # ImageReader(module='imageio')
        widget.setImagelike('https://upload.wikimedia.org/wikipedia/commons/'
                            'thumb/b/b4/Last_Supper_by_Leonardo_da_Vinci.jpg/'
                            '320px-Last_Supper_by_Leonardo_da_Vinci.jpg')
        self.setWidget(widget)

    @runnable
    @staticmethod
    def demoQLayerSelector() -> Network:
        """Demonstration of the :py:class:`QLayerSelector`.
        """
        return Network['alexnet-tf']

    def demoFinishQLayerSelector(self, network: Network) -> None:
        """Demonstration of the :py:class:`QLayerSelector`.
        """
        widget = QLayerSelector()
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

    def demoQTrainingBox(self) -> None:
        """Demonstration of the :py:class:`QTrainingBox`.
        """
        trainer = Trainer()
        self.setWidget(QTrainingBox(trainer=trainer, epochs=5))
        # self.demoPrepareQTrainingBox1(trainer)
        self.prepareQTrainingBox2(trainer, run=True)

    @runnable
    def prepareQTrainingBox1(self, trainer: Trainer) -> None:
        """Prepare the trainer by providing a trainee and training data.
        """
        trainer.trainee = MockupTrainee()
        trainer.training_data = Noise(length=1000)

    @runnable
    def prepareQTrainingBox2(self, trainer: Trainer) -> None:
        """Prepare the trainer by providing a trainee and training data.
        """
        print("importing packages for training ...")
        from dltb.thirdparty.tensorflow.ae import Autoencoder
        print("... import done - creating objects  ...")

        datasource = Datasource(module='mnist', one_hot=True)
        trainer.training_data = datasource
        trainer.trainee = Autoencoder(shape=datasource.shape, code_dim=2)

        print("... finished!")


if __name__ == '__main__':
    demo = QDemo()
    demo.show()
    result = debug.application.exec_()
