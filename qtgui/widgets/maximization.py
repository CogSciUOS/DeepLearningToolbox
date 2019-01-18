# FIXME[todo]
#    _LARGER_IMAGE: bool
#        now called upscaling in thesis
#    _IMAGE_SIZE_X: int
#    _IMAGE_SIZE_Y: int
#
#    _JITTER: bool
#    _JITTER_STRENGTH: int
#
#    _WRAP_AROUND: bool

# FIXME[ideas]:
#  * not only run/stop but also step to inspect intermediate results

"""
This module offers a hub to select all important constants used in the
AlexNet activation maximization module.

File: maximization.py
Author: Antonia Hain, Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import copy

from controller import (BaseController,
                        ActivationsController, MaximizationController)
from model import Model, ModelObserver, ModelChange
from qtgui.utils import QImageView, QLogHandler
from qtgui.widgets import QNetworkSelector


from PyQt5.QtCore import Qt

from PyQt5.QtGui import QFontMetrics, QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import (QWidget, QLabel, QCheckBox, QLineEdit,
                             QComboBox, QPushButton,
                             QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QSizePolicy)

import cv2

from .matplotlib import QMatplotlib
from tools.am import Config, ConfigObserver, EngineObserver


class QMaximizationConfig(QWidget, ModelObserver, EngineObserver, ConfigObserver):
    """
    A widget to displaying controls for the parameters of the
    activation maximization visualization algorithm.

    Attributes
    ----------
    _config: Config
        The Config object currently controlled by this QMaximizationConfig
        widget. None if no config object is controlled.
        (we need this basically to be able to stop observing it)

    _network: Network
        (we need this basically to obtain the number of units in a layer)

    _networkSelector: QNetworkSelector
    _layerSelector: QComboBox
        The currently selected layer
        (corresponds to Config.LAYER_KEY)
    _unitIndex: QLineEdit
    _buttonRandomUnit: QCheckBox
        randomly select a unit in the layer.

    _eta: QLineEdit
    _checkRandomizeInput: QCheckBox

    _checkBlur: QGroupBox
    _blurKernelWidth: QLineEdit
    _blurKernelHeight: QLineEdit
    _blurKernelSigma: QLineEdit
    _blurKernelFrequency: QLineEdit

    _checkL2Activated: QGroupBox
    _l2Lambda: QLineEdit

    _checkNormClipping: QGroupBox
    _normClippingFrequency: QLineEdit
    _normPercentile: QLineEdit

    _checkContributionClipping: QGroupBox
    _contributionClippingFrequency: QLineEdit
    _contributionPercentile = QLineEdit()

    _checkBorderReg: QGroupBox
    _borderFactor: QLineEdit
    _borderExp: QLineEdit

    _largerImage: QGroupBox
    _imageSizeX: QLineEdit
    _imageSizeY: QLineEdit

    _jitter: QGroupBox
    _jitterStrength: QLineEdit

    _checkWrapAround: QCheckBox

    _lossGoal: QLineEdit
    _lossCount: QLineEdit
    _maxSteps: QLineEdit

    _checkTensorboard: QCheckBox
    _checkNormalizeOutput: QCheckBox
    """

    def __init__(self, parent=None):
        """Initialization of the ActivationsPael.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        model   :   model.Model
                    The backing model. Communication will be handled by a
                    controller.
        """
        super().__init__(parent)
        self._widgets = {}
        self._config = None
        self._initUI()
        self._layoutComponents()

    def _initUI(self):
        """Add additional UI elements.

            * The ``QActivationView`` showing the unit activations on the left

        """
        # Some general remarks on QLineEdit:
        #   - textChanged(text) is emited whenever the contents of the
        #     widget changes
        #   - textEdited(text) is emited only when the user changes the
        #     text using mouse and keyboard (so it is not emitted when
        #     QLineEdit::setText() is called).
        #   - editingFinished() only is emmitted when inputMask and
        #     Validator are ok


        #
        #  Unit selection
        #

        # FIXME[concept]: probably we want the network to be selected
        #  globally (i.e. same in "Activations" and "Maximization" panel),
        #  while the Layer/Unit may be local, (i.e. different in
        #  "Activations" and "Maximization" panel).

        self._networkSelector = QNetworkSelector()
        
        # in which layer the unit is found. needs to be a key in layer_dict in
        # am_alexnet (LAYER_KEY)
        self._layerSelector = QComboBox()
        self._layerSelector.setToolTip("Select layer for visualization")
        def slot(text):
            if self._config is not None:
                self._config.LAYER_KEY = text
        self._layerSelector.currentIndexChanged[str].connect(slot)

        # select a random unit from the chosen layer to perform activation
        # maximization
        self._buttonRandomUnit = QPushButton("Random unit")
        self._buttonRandomUnit.setToolTip("Choose a random unit "
                                          "for visualization")
        def slot():
            if self._config is not None:
                maxUnit = self._maxUnit()
                if maxUnit is not None:
                    self._config.UNIT_INDEX = np.random.randint(0, maxUnit)
        self._buttonRandomUnit.clicked.connect(slot)

        # _unitIndex: A text field to manually enter the index of
        # desired input (UNIT_INDEX).
        self._unitIndex = QLineEdit()
        self._unitIndex.setMaxLength(5)
        self._unitIndex.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self._unitIndex.setMinimumWidth(
            QFontMetrics(self.font()).width('5') * 4)
        self._connect_to_config(self._unitIndex, 'UNIT_INDEX')

        self._unitMax = QLabel()
        self._unitMax.setMinimumWidth(
            QFontMetrics(self.font()).width('8') * 4)
        self._unitMax.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Expanding)

        #
        # Maximization parameters
        #

        # learning rate
        self._eta = QLineEdit()
        self._eta.setMaxLength(8)
        self._eta.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self._eta.setMinimumWidth(QFontMetrics(self.font()).width('8') * 4)
        self._connect_to_config(self._eta, 'ETA')

        # whether maximization is initialized with random input or flat
        # colored image (RANDOMIZE_INPUT)
        self._checkRandomizeInput = QCheckBox("Initialize Input Random")
        self._connect_to_config(self._checkRandomizeInput, 'RANDOMIZE_INPUT')

        #
        # Gaussian blur parameters
        #

        # BLUR_ACTIVATED = True
        self._checkBlur = QGroupBox("Blur")
        self._checkBlur.setCheckable(True)
        self._connect_to_config(self._checkBlur, 'BLUR_ACTIVATED')

        # BLUR_KERNEL = (3, 3)
        self._blurKernelWidth = QLineEdit()
        self._blurKernelHeight = QLineEdit()
        for w in self._blurKernelWidth, self._blurKernelHeight:
            w.setMaxLength(2)
        self._connect_to_config((self._blurKernelWidth,
                                 self._blurKernelHeight), 'BLUR_KERNEL')

        # BLUR_SIGMA = 0
        self._blurKernelSigma = QLineEdit()
        self._connect_to_config(self._blurKernelSigma, 'BLUR_SIGMA')

        # BLUR_FREQUENCY = 5  # how many steps between two blurs. paper used 4.
        self._blurKernelFrequency = QLineEdit()
        self._connect_to_config(self._blurKernelFrequency, 'BLUR_FREQUENCY')

        #
        # l2 decay parameters
        #

        # L2_ACTIVATED = True
        self._checkL2Activated = QGroupBox("L2 regularization")
        self._checkL2Activated.setCheckable(True)
        self._connect_to_config(self._checkL2Activated, 'L2_ACTIVATED')

        # L2_LAMBDA = 0.000001  # totally arbitrarily chosen
        self._l2Lambda = QLineEdit()
        self._l2Lambda.setMaxLength(10)
        self._connect_to_config(self._l2Lambda, 'L2_LAMBDA')

        #
        # low norm pixel clipping parameters
        #

        # NORM_CLIPPING_ACTIVATED = False
        self._checkNormClipping = QGroupBox("Norm Clipping")
        self._checkNormClipping.setCheckable(True)
        self._connect_to_config(self._checkNormClipping,
                                'NORM_CLIPPING_ACTIVATED')

        # NORM_CLIPPING_FREQUENCY: how many steps between pixel clippings
        self._normClippingFrequency = QLineEdit()
        self._normClippingFrequency.setMaxLength(10)
        self._connect_to_config(self._normClippingFrequency,
                                'NORM_CLIPPING_FREQUENCY')

        # NORM_PERCENTILE = 30  # how many of the pixels are clipped
        self._normPercentile = QLineEdit()
        self._normPercentile.setMaxLength(10)
        self._connect_to_config(self._normPercentile,
                                'NORM_PERCENTILE')

        # low contribution pixel clipping parameters
        # see norm clipping for explanation
        # CONTRIBUTION_CLIPPING_ACTIVATED = True
        self._checkContributionClipping = QGroupBox("Contribution Clipping")
        self._checkContributionClipping.setCheckable(True)
        self._connect_to_config(self._checkContributionClipping,
                                'CONTRIBUTION_CLIPPING_ACTIVATED')

        # CONTRIBUTION_CLIPPING_FREQUENCY = 50
        self._contributionClippingFrequency = QLineEdit()
        self._contributionClippingFrequency.setMaxLength(10)
        self._connect_to_config(self._contributionClippingFrequency,
                                'CONTRIBUTION_CLIPPING_FREQUENCY')
        # CONTRIBUTION_PERCENTILE = 15
        self._contributionPercentile = QLineEdit()
        self._contributionPercentile.setMaxLength(10)
        self._connect_to_config(self._contributionPercentile,
                                'CONTRIBUTION_PERCENTILE')

        # border regularizer - punishes pixel values the higher their
        # distance to the image center
        # BORDER_REG_ACTIVATED = True
        self._checkBorderReg = QGroupBox("Border Regularizer")
        self._checkBorderReg.setCheckable(True)
        self._connect_to_config(self._checkBorderReg, 'BORDER_REG_ACTIVATED')
        # instead of summing up the product of the actual pixel to
        # center distance and its value (too strong), the effect is
        # changed by multiplying each of the resultant values with
        # this factor
        # BORDER_FACTOR = 0.000003
        self._borderFactor = QLineEdit()
        self._borderFactor.setMaxLength(10)
        self._connect_to_config(self._borderFactor, 'BORDER_FACTOR')

        self._borderExp = QLineEdit()
        self._borderExp.setMaxLength(10)
        self._connect_to_config(self._borderExp, 'BORDER_EXP')

        #
        # larger image
        #
        self._largerImage = QGroupBox("Larger Image")
        self._largerImage.setCheckable(True)
        self._connect_to_config(self._largerImage, 'LARGER_IMAGE')

        self._imageSizeX = QLineEdit()
        self._imageSizeX.setMaxLength(4)
        self._connect_to_config(self._imageSizeX, 'IMAGE_SIZE_X')
        
        self._imageSizeY = QLineEdit()
        self._imageSizeY.setMaxLength(4)
        self._connect_to_config(self._imageSizeY, 'IMAGE_SIZE_Y')

        self._jitter = QGroupBox("Jitter")
        self._jitter.setCheckable(True)
        self._connect_to_config(self._jitter, 'JITTER')

        self._jitterStrength = QLineEdit()
        self._jitterStrength.setMaxLength(4)
        self._connect_to_config(self._jitterStrength, 'JITTER_STRENGTH')

        self._checkWrapAround = QCheckBox("Wrap around")
        self._checkWrapAround.setEnabled(False) # FIXME[todo]: disabling WrapAround not implemented yet
        self._connect_to_config(self._checkWrapAround, 'WRAP_AROUND')

        # convergence parameters
        # relative(!) difference between loss and last 50 losses to converge to
        # LOSS_GOAL = 0.01
        self._lossGoal = QLineEdit()
        self._lossGoal.setMaxLength(5)
        self._connect_to_config(self._lossGoal, 'LOSS_GOAL')

        self._lossCount = QLineEdit()
        self._lossCount.setMaxLength(5)
        self._connect_to_config(self._lossCount, 'LOSS_COUNT')


        # MAX_STEPS = 2000
        # how many steps to maximally take when optimizing an image
        self._maxSteps = QLineEdit()
        self._maxSteps.setMaxLength(5)
        self._connect_to_config(self._maxSteps, 'MAX_STEPS')

        # to easily switch on and off the logging of the image and loss
        # TENSORBOARD_ACTIVATED = False
        self._checkTensorboard = QCheckBox("Tensorboard output")
        self._connect_to_config(self._checkTensorboard,
                                'TENSORBOARD_ACTIVATED')

        # whether to save output image normalized
        # NORMALIZE_OUTPUT = True
        self._checkNormalizeOutput = QCheckBox("Normalize Output")
        self._connect_to_config(self._checkNormalizeOutput, 'NORMALIZE_OUTPUT')



    def _layoutComponents(self):
        """Arrange the components of this QConfig.
        """
        boxes = []

        #
        # Column 1
        #
        
        box = QGroupBox('Layer and Unit')
        boxes.append(box)

        unitLayout = QVBoxLayout(box)
        networkBox = QHBoxLayout()
        networkBox.addWidget(QLabel("Network: "))
        networkBox.addWidget(self._networkSelector)
        unitLayout.addLayout(networkBox)

        layerBox = QHBoxLayout()
        layerBox.addWidget(QLabel("Layer: "))
        layerBox.addWidget(self._layerSelector)
        unitLayout.addLayout(layerBox)
        
        l = QHBoxLayout()
        l.addWidget(QLabel("Unit: "))
        l.addWidget(self._unitIndex)
        l.addWidget(QLabel("/"))
        l.addWidget(self._unitMax)
        unitLayout.addLayout(l)
        unitLayout.addWidget(self._buttonRandomUnit)
        etaBox = QHBoxLayout()
        etaBox.addWidget(QLabel("Eta"))
        etaBox.addWidget(self._eta)
        unitLayout.addLayout(etaBox)
        unitLayout.addWidget(self._checkRandomizeInput)
        unitLayout.addStretch(1)

        #
        # Blur
        #
        blurLayout = QVBoxLayout()
        self._checkBlur.setLayout(blurLayout)
        
        blurKernel = QHBoxLayout()
        blurKernel.addWidget(QLabel("Kernel size: "))
        blurKernel.addStretch()
        blurKernel.addWidget(self._blurKernelWidth)
        blurKernel.addWidget(QLabel("x"))
        blurKernel.addWidget(self._blurKernelHeight)
        blurLayout.addLayout(blurKernel)
        
        blurParam = QHBoxLayout()
        blurParam.addWidget(QLabel("Sigma: "))
        blurParam.addWidget(self._blurKernelSigma)
        blurLayout.addLayout(blurParam)      

        blurParam = QHBoxLayout()
        blurParam.addWidget(QLabel("Frequency: "))
        blurParam.addWidget(self._blurKernelFrequency)
        blurLayout.addLayout(blurParam)      

        #
        # Larger image
        #
        largeImageLayout = QVBoxLayout()
        largeImageSize = QHBoxLayout()
        largeImageSize.addWidget(QLabel("Image size: "))
        largeImageSize.addStretch()
        largeImageSize.addWidget(self._imageSizeX)
        largeImageSize.addWidget(QLabel("x"))
        largeImageSize.addWidget(self._imageSizeY)
        largeImageLayout.addLayout(largeImageSize)
        self._largerImage.setLayout(largeImageLayout)

        #
        # Jitter
        #
        jitterStrengthLayout = QHBoxLayout()
        jitterStrengthLayout.addWidget(QLabel("Strength: "))
        jitterStrengthLayout.addStretch()
        jitterStrengthLayout.addWidget(self._jitterStrength)
        self._jitter.setLayout(jitterStrengthLayout)


        #
        # L2 regularization
        #
        l2Layout = QVBoxLayout()
        l2Param = QHBoxLayout()
        l2Param.addWidget(QLabel("Lambda: "))
        l2Param.addWidget(self._l2Lambda)
        l2Layout.addLayout(l2Param)
        self._checkL2Activated.setLayout(l2Layout)


        #
        # contribution clipping
        #      
        normLayout = QVBoxLayout()

        normParam1 = QHBoxLayout()
        normParam1.addWidget(QLabel("Frequncy: "))
        normParam1.addWidget(self._normClippingFrequency)
        normLayout.addLayout(normParam1)
        
        normParam2 = QHBoxLayout()
        normParam2.addWidget(QLabel("Percentile: "))
        normParam2.addWidget(self._normPercentile)
        normLayout.addLayout(normParam2)

        self._checkNormClipping.setLayout(normLayout)

        #
        # contribution clipping
        #      
        contribParams = QVBoxLayout()

        contribParam1 = QHBoxLayout()
        contribParam1.addWidget(QLabel("Frequncy: "))
        contribParam1.addWidget(self._contributionClippingFrequency)
        contribParams.addLayout(contribParam1)
        
        contribParam2 = QHBoxLayout()
        contribParam2.addWidget(QLabel("Percentile: "))
        contribParam2.addWidget(self._contributionPercentile)
        contribParams.addLayout(contribParam2)
        self._checkContributionClipping.setLayout(contribParams)

        #
        # border regularization
        #      
        borderParams = QVBoxLayout()
        
        borderParam = QHBoxLayout()
        borderParam.addWidget(QLabel("Factor: "))
        borderParam.addWidget(self._borderFactor)
        borderParams.addLayout(borderParam)
        
        borderParam = QHBoxLayout()
        borderParam.addWidget(QLabel("Exponent: "))
        borderParam.addWidget(self._borderExp)
        borderParams.addLayout(borderParam)
        self._checkBorderReg.setLayout(borderParams)

        #
        # main layout
        #

        box2 = QVBoxLayout()
        box2.addWidget(self._checkL2Activated)
        box2.addWidget(self._checkBorderReg)
        box2.addStretch()

        box3 = QVBoxLayout()
        box3.addWidget(self._checkBlur)
        box3.addWidget(self._checkNormClipping)
        box3.addWidget(self._checkContributionClipping)
        box3.addStretch()
        
        box4 = QVBoxLayout()
        box4.addWidget(self._largerImage)
        box4.addWidget(self._jitter)
        box4.addWidget(self._checkWrapAround)
        box4.addStretch()

        box5 = QGroupBox('Optimization')
        box5layout = QVBoxLayout()

        box5Param1 = QHBoxLayout()
        box5Param1.addWidget(QLabel("Loss goal: "))
        box5Param1.addWidget(self._lossGoal)
        box5layout.addLayout(box5Param1)

        l = QHBoxLayout()
        l.addWidget(QLabel("Loss count: "))
        l.addWidget(self._lossCount)
        box5layout.addLayout(l)
        
        box5Param2 = QHBoxLayout()
        box5Param2.addWidget(QLabel("Max steps: "))
        box5Param2.addWidget(self._maxSteps)
        box5layout.addLayout(box5Param2)


        box5layout.addWidget(self._checkTensorboard)
        box5layout.addWidget(self._checkNormalizeOutput)
        box5layout.addStretch()
        box5.setLayout(box5layout)

        boxLayout = QHBoxLayout()
        boxLayout.addWidget(box)
        boxLayout.addLayout(box2)
        boxLayout.addLayout(box3)
        boxLayout.addLayout(box4)
        boxLayout.addWidget(box5)
        self.setLayout(boxLayout)

    def setConfig(self, config: Config) -> None:
        """Set the Config object to be controlled by this
        QMaximizationConfig. The values of that object will be
        displayed in this QMaximizationConfig, and any changes
        entered will be sent to that Config object.

        Parameters
        ----------
        config: Config
            The Config object or None to decouple this
            QMaximizationConfig object (FIXME[todo]: all input Widgets
            will be cleared and disabled).
        """
        
        if self._config is not None:
            self._config.remove_observer(self)
        self._config = config

        if self._config is not None:
            self._config.addObserver(self, notify=True)
            
    def setFromConfigView(self, configView: 'QMaximizationConfigView'):
        self.setConfig(configView.config)

    def _connect_to_config(self, widget, attr):
        self._widgets[attr] = widget
        if isinstance(widget, QWidget):
            widget.setToolTip(Config._config.get(attr).get('doc'))
        if isinstance(widget, QLineEdit):
            widget.setAlignment(Qt.AlignRight)
            value = Config._config.get(attr).get('default')
            if isinstance(value, int):
                widget.setValidator(QIntValidator())
                def slot(text):
                    if self._config is not None:
                        try: value = int(text)
                        except ValueError: value = 0
                        setattr(self._config, attr, value)
            else:
                widget.setValidator(QDoubleValidator())
                def slot(text):
                    if self._config is not None:
                        try: value = float(text)
                        except ValueError: value = 0
                        setattr(self._config, attr, value)
            widget.textEdited.connect(slot)
        elif isinstance(widget, tuple):
            def slot(text):
                if self._config is not None:
                    value = ()
                    for w in widget:
                        try: value += int(text)
                        except ValueError: value += 0
                    setattr(self._config, attr, value)
            for w in widget:
                w.setToolTip(Config._config.get(attr).get('doc'))
                w.setAlignment(Qt.AlignRight)
                w.setValidator(QIntValidator())
                w.textEdited.connect(slot)
        elif isinstance(widget, QGroupBox):
            def slot(state):
                if self._config is not None:
                    setattr(self._config, attr, bool(state))
            widget.toggled.connect(slot)
        elif isinstance(widget, QCheckBox):
            def slot(state):
                if self._config is not None:
                    setattr(self._config, attr, bool(state))
            widget.stateChanged.connect(slot)

        
    def _maxUnit(self, layer_id=None) -> int:
        if layer_id is None:
            layer_id = self._config.LAYER_KEY
        if layer_id is None:
            return None

        network = self._networkSelector.network
        if network is None:
            return None
        value = network.get_layer_output_units(layer_id)
        return value

    def setController(self, controller: BaseController):
        logger.info(f"!!! QMaximizationConfig: set Controller of type {type(controller)}")
        if isinstance(controller, ActivationsController):
            super().setController(controller)
            # FIXME[concepts]: we do not want to change the global layer, do we?
            #self._layerSelector.currentIndexChanged[str].\
            #    connect(controller.onLayerSelected)
            # FIXME[todo]: but we want to change the global network!
        elif isinstance(controller, MaximizationController):
            #super().setController(controller, '_maximization_controller')
            engine = controller.get_engine()
            self.setConfig(engine.config)


    def modelChanged(self, model: ModelObserver, info: ModelChange) -> None:
        """The model has changed. Here we only react to changes of the
        network. We will not react to changes of the model layer as we
        will use our own layer.
        """

        if info.network_changed:
            network = model.network
            self._layerSelector.clear()
            if network is not None:
                self._layerSelector.addItems(network.layer_dict.keys())
                self._config.NETWORK_KEY = network.get_id()
                # Default strategy: select the last layer:
                self._config.LAYER_KEY = network.output_layer_id()

    def configChanged(self, config: Config, info: Config.Change) -> None:
        """Handle a change in configuration.
        """
        """Update the values displayed in this QMaximizationConfig from
        the given Config object.

        Parameters
        ----------
        config: Config
        """
        #
        # update the _layerSelector
        #

        for name in self._widgets.keys():
            widget = self._widgets[name]
            if isinstance(widget, tuple):
                for i,w in enumerate(widget):
                    w.setText(str(getattr(config, name)[i]))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(getattr(config, name)))
            elif isinstance(widget, QGroupBox):
                widget.setChecked(getattr(config, name))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(getattr(config, name))

        if info.layer_changed:
            if self._layerSelector.findText(config.LAYER_KEY) != -1:
                self._layerSelector.setCurrentText(config.LAYER_KEY)
                maxUnit = self._maxUnit(config.LAYER_KEY)
                if maxUnit is None:
                    self._unitMax.setText("*")
                else:
                    self._unitMax.setText(str(maxUnit))
            else:
                self._unitMax.setText("*")



from PyQt5.QtCore import pyqtSignal

class QMaximizationConfigView(QWidget):
    """
    
    Signals
    -------
    selected
        The 'selected' signal is emitted when this
        QMaximizationConfigView was selected.
    """

    clicked = pyqtSignal(object)
    
    selected = pyqtSignal(object)

    def __init__(self, parent=None, config:Config=None):
        super().__init__(parent)
        self._label = QLabel()
        self._config = None
        self.config = config
        self._initUI()

    def _initUI(self):
        layout = QVBoxLayout()
        layout.addWidget(self._label)
        self.setLayout(layout)

    @property
    def config(self) -> Config:
        return self._config

    @config.setter
    def config(self, config: Config) -> None:
        #if self._config is not None:
        #    self._config.remove_observer(self)

        if config is not None:
            self._config = copy.copy(config)
        else:
            self._config = None
        if self._config is not None:
            #self._config.addObserver(self)
            string = ("Layer: <b>" + str(config.LAYER_KEY) + "</b>, " +
                      "unit: <b>" + str(config.UNIT_INDEX) + "</b>")
        else:
            string = None
        self._label.setText(string)

    def setFromEngine(self, engine):
        self.config = engine.config 

    def mousePressEvent(self, event):
        """Process mouse event. A mouse click will select this
        QMaximizationConfigView, emitting the 'clicked' signal.

        Parameters
        ----------
        event : QMouseEvent
        """
        self.clicked.emit(self)

    def mouseDoubleClickEvent(self, event):
        """Process mouse event. A mouse click will select this
        QMaximizationConfigView, emitting the 'clicked' signal.

        Parameters
        ----------
        event : QMouseEvent
        """
        self.select()

    def select(self):
        self.selected.emit(self)


from PyQt5.QtWidgets import QSlider

import os

from tools.am import Engine, EngineObserver, EngineChange
from controller import MaximizationController

import numpy as np
import tensorflow as tf

from util import async


# FIXME[concept]: currently the observer concept does not support
# dealing with multiple controllers: there is just one self._controller
# variable. Each Controller is associate with one type of Observer.
# This may be too restrictive: we may want to contact multiple Controllers!
class QMaximizationControls(QWidget, ModelObserver, EngineObserver):
    """
    Attributes
    ----------
    _imageView: QImageView

    _engine: Engine

    _maximization_controller: MaximizationController

    display: QMaximizationDisplay
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._stack = None
        self._stack_grow = False
        self._stack_size = 0
        self._losses = []
        self._maximization_controller = None
        self._initUI()
        self.display = None

    def _initUI(self):
        
        self._button_run = QPushButton("run")
        self._button_run.clicked.connect(self.onMaximize)

        self._button_stop = QPushButton("stop")
        self._button_stop.clicked.connect(self.onStop)
        self._button_stop.setEnabled(False)

        self._button_show = QPushButton("show")
        self._button_show.clicked.connect(self.onShow)

        self._button_reset = QPushButton("reset")
        self._button_reset.clicked.connect(self.onReset)
        
        self._checkbox_record = QCheckBox("Record images")
        self._info_record = QLabel()
        self._button_record_save = QPushButton("Save")
        self._button_record_save.clicked.connect(self.onSaveMovie)
        self._button_record_save.setEnabled(False)

        self._imageView = QImageView()

        self._info = QLabel()
        self._info_iteration = QLabel()
        self._info_loss = QLabel()
        self._info_minmax = QLabel()
        self._info_mean = QLabel()

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setEnabled(self._losses is not None)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setSliderPosition(0)
        self._slider.valueChanged.connect(self.selectIteration)

        self._logWindow = QLogHandler()
        self._logWindow.setLevel(logging.DEBUG)
        self._logWindow.setMinimumWidth(600)
        logger.handlers = []
        logger.addHandler(self._logWindow)

        self._plt = QMatplotlib()
        
        self._layoutComponents()

    def _layoutComponents(self):
        info = QVBoxLayout()
        buttons = QHBoxLayout()
        buttons.addWidget(self._button_run)
        buttons.addWidget(self._button_stop)
        buttons.addWidget(self._button_show)
        buttons.addWidget(self._button_reset)
        info.addLayout(buttons)
        info_state = QHBoxLayout()
        info_state.addWidget(QLabel("Engine:"))
        info_state.addWidget(self._info)
        info.addLayout(info_state)
        info_iteration = QHBoxLayout()
        info_iteration.addWidget(QLabel("Iteration:"))
        info_iteration.addWidget(self._info_iteration)
        info.addLayout(info_iteration)
        info_loss = QHBoxLayout()
        info_loss.addWidget(QLabel("Loss:"))
        info_loss.addWidget(self._info_loss)
        info.addLayout(info_loss)
        info_minmax = QHBoxLayout()
        info_minmax.addWidget(QLabel("Min/max:"))
        info_minmax.addWidget(self._info_minmax)
        info.addLayout(info_minmax)
        info_mean = QHBoxLayout()
        info_mean.addWidget(QLabel("Mean:"))
        info_mean.addWidget(self._info_mean)
        info.addLayout(info_mean)
        info.addWidget(self._slider)
        record_box = QHBoxLayout()
        record_box.addWidget(self._checkbox_record)
        record_box.addWidget(self._info_record)
        record_box.addWidget(self._button_record_save)
        record_box.addStretch()
        info.addLayout(record_box)
        info.addStretch()
        
        imageView = QVBoxLayout()
        imageView.addWidget(self._imageView)
        imageView.addStretch()

        layout = QHBoxLayout()
        layout.addLayout(info)        
        layout.addLayout(imageView)
        layout.addWidget(self._plt)
        layout.addWidget(self._logWindow)
        layout.addStretch()
        
        self.setLayout(layout)

    @property
    def engine(self) -> Engine:
        return self._engine

    @engine.setter
    def engine(self, engine: Engine) -> None:
        logger.info(f"!!! QMaximizationControls.set_engine: ({type(engine)}) !!!")
        # FIXME[problem]: it seems not to be possible to log from another
        # thread: This r
        import tools.am.engine
        tools.am.engine.logger.addHandler(self._logWindow)
        self._engine = engine

    @property
    def config(self) -> Config:
        return self._config

    @config.setter
    def config(self, config: Config) -> None:
        logger.info(f"!!! QMaximizationControls.set_config: ({type(config)}) !!!")
        self._config = config

    def onStop(self):
        """Stop a running computation.
        """
        if self._maximization_controller is not None:
            self._maximization_controller.onStop()

    def setController(self, controller: BaseController):
        logger.info(f"!!! QMaximizationControls: set Controller of type {type(controller)}")
        if isinstance(controller, ActivationsController):
            super().setController(controller)
        elif isinstance(controller, MaximizationController):
            super().setController(controller, '_maximization_controller')
            self.engine = controller.get_engine()
            self.config = self._engine.config


    def modelChanged(self, model: ModelObserver, info: ModelChange) -> None:

        network = model.network

        logger.info(f"{self.__class__.__name__}.modelChanged(): {network}")

        if info.network_changed:
            if network is not None:
                self._engine.network = network
                self._image_shape = network.get_input_shape(False)
                logger.info(f"input shape: {self._image_shape} ({type(self._image_shape)})")

    def engineChanged(self, engine: 'Engine', info: EngineChange) -> None:

        if info.network_changed:
            logger.info("!!! QMaximizationControls: network changed")

        if info.config_changed:
            logger.info("!!! QMaximizationControls: config changed")

        if info.engine_changed:
            logger.info("!!! QMaximizationControls: engine changed")
            self._info.setText(engine.status)
            self._info.setText(engine.status)
            self._button_run.setEnabled(not engine.running)
            self._button_stop.setEnabled(engine.running)
            self._button_show.setEnabled(not engine.running)
            self._button_reset.setEnabled(not engine.running)
            self._button_record_save.setEnabled(self._stack is not None)

            if not engine.running and engine._iteration > 0:  # FIXME[hack]: private variable
                image_normalized = self._normalizeImage(engine.image)
                self._imageView.setImage(image_normalized)
                if self.display is not None:
                    self.display.showImage(image_normalized, engine)

        if info.image_changed:
            logger.info("!!! QMaximizationControls: image changed")

            iteration = engine._iteration
            self._info_iteration.setText(str(iteration))
            if iteration >= 0:
                self._info_loss.setText(f"{engine._loss[iteration-1]:.2f}")

                self._plt._ax.clear()
                self._plt._ax.plot(np.arange(iteration),
                                   engine._loss[:iteration])
                self._plt._ax.figure.canvas.draw()

                self._slider.setMaximum(iteration)
                # FIXME[concept]: this triggers the valueChanged signal,
                # i.e. it will call selectIteration
                #self._slider.setSliderPosition(iteration)

            self._showImage(engine.image)



    def _normalizeImage(self, image: np.ndarray,
                        as_uint8:bool = True,
                        as_bgr = False) -> np.ndarray:
        # FIXME[design]: this put be done somewhere else ...
        normalized = np.ndarray(image.shape, image.dtype)
        cv2.normalize(image, normalized, 0, 255, cv2.NORM_MINMAX)
        # self.image_normalized = (self.image-min_value)*255/(max_value-min_value)
        if as_uint8:
            normalized = normalized.astype(np.uint8)
        if as_bgr:
            normalized = normalized[...,::-1]
        return normalized

    def onMaximize(self):
        """
        """
        logger.info("QMaximizationControls.onMaximize() -- begin")
        self._maximization_controller.onMaximize()
        logger.info("QMaximizationControls.onMaximize() -- end")

    def onShow(self):
        """Respond to the 'show' button.  Set the current image as
        input data to the controller.  This causes the activations
        (and the classification result) to be displayed in the
        'Activations' panel.
        """
        self._controller.onNewInput(self._imageView.getImage(),
                                    self._config.UNIT_INDEX,
                                    self._engine.description)

    def onReset(self):
        """Respond to the 'reset' button. This will reset the engine
        and start a new maximization run.
        """

        # FIXME[concept]: this should be done by the Engine.
        if self._checkbox_record.checkState():
            if self._stack_grow:
                self._stack = np.ndarray((0,) + tuple(self._image_shape))
            else:
                self._stack = np.ndarray((self._config.MAX_STEPS+1,) +
                                         tuple(self._image_shape))
        else:
            self._stack = None
        self._stack_size = 0
        
        # FIXME[concept]: this should be done in response to engine events
        self._button_record_save.setEnabled(False)

        # FIXME[concept]: this should be done in response to engine events
        #self._button_run.setEnabled(False)
        #self._button_stop.setEnabled(True)
        #self._button_show.setEnabled(False)

        # FIXME[concept]: this should be done in response to engine events
        self._info_iteration.setText("")
        self._info_loss.setText("")
        self._info_minmax.setText("")
        self._info_mean.setText("")
        self._info_record.setText("")
        self._losses = []
        self._slider.setEnabled(self._losses is not None)
        logger.info("QMaximizationControls.onMaximize(True) -- begin")
        self._maximization_controller.onMaximize(True)
        logger.info("QMaximizationControls.onMaximize() -- end")

    def onSaveMovie(self):
        # http://www.fourcc.org/codecs.php
        #fourcc, suffix = 'PIM1', 'avi'
        #fourcc, suffix = 'ffds', 'mp4'
        fourcc, suffix = 'MJPG', 'avi' # https://en.wikipedia.org/wiki/Motion_JPEG
        #fourcc, suffix = 'XVID', 'avi'

        filename = 'activation_maximization.' + suffix
        fps = 25
        frameSize = self._stack.shape[1:3]
        isColor = (self._stack.ndim==4)
        logger.info(f"onSaveMovie: preparing {filename} ({fourcc}) ... ")
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*fourcc),
                                 fps, frameSize, isColor=isColor)
        logger.info(f"onSaveMovie: writer.isOpened: {writer.isOpened()}")
        if writer.isOpened():
            logger.info(f"onSaveMovie: writing {len(self._stack_size)} frames ... ")
            # FIXME[todo]: we need the number of actual frames!
            for frame in range(self._stack_size):
                writer.write(self._normalizeImage(self._stack[frame],
                                                  as_bgr=True))
        logger.info(f"onSaveMovie: releasing the writer")
        writer.release()
        logger.info(f"onSaveMovie: done")

    #@async
    # FIXME[old]
    def logOptimizationStep(self, image: np.ndarray,
                            iteration: int, loss: float):
        self._losses.append(loss) 
        if self._stack is not None:
            if self._stack_grow:
                self._stack = np.concatenate((self._stack, image[np.newaxis]),
                                             axis=0)
            else:
                self._stack[iteration] = image
            self._stack_size += 1
            # 10^6 Bytes = 1 MB, 2^20 Bytes = 1 MiB
            self._info_record.setText("(memory={:,}/{:,} MiB)".
                                      format((self._stack[:self._stack_size].nbytes) >> 20,
                                             self._stack.nbytes >> 20))
        self._showImage(image)
        self._plt._ax.clear()
        self._plt._ax.plot(np.arange(iteration+1),
                           self._engine._loss[:iteration+1])
        self._plt._ax.figure.canvas.draw()

        self._slider.setMaximum(len(self._losses)-1)
        # FIXME[concept]: this triggers the valueChanged signal,
        # i.e. it will call selectIteration
        self._slider.setSliderPosition(len(self._losses)-1)

        self.update()

    def _showImage(self, image: np.ndarray):
        if image is not None:
            image_min = image.min()
            image_max = image.max()
            image_mean = image.mean()
            image_std = image.std()
        else:
            image_min = 0
            image_max = 0
            image_mean = 0
            image_std = 0
        self._info_minmax.setText(f"{image_min:.2f}/{image_max:.2f}"
                                  f" ({image_max-image_min:.2f})")
        self._info_mean.setText(f"{image_mean:.2f} +/- {image_std:.2f}")

        if image is not None:
            self._imageView.setImage(self._normalizeImage(image))

        
    def selectIteration(self, iteration: int) -> None:
        self._info_iteration.setText(f"{iteration}")
        #self._info_loss.setText(f"{self._losses[iteration]:.2f}")
        #if self._stack is not None:
        #    self._showImage(self._stack[iteration])
        self._slider.setSliderPosition(iteration)
        self.update()
    

from PyQt5.QtWidgets import QFrame

import numpy as np

class QMaximizationDisplay(QWidget):
    """A Widget to display the result of activation maximization. The
    widget displays a grid allowing to display different results next
    to each other.

    Attributes
    ----------
    """

    def __init__(self, parent=None, rows:int=2, columns:int=5):
        super().__init__(parent)
        self._index = None
        self._boxes = []
        self._imageView = []
        self._configView = []
        self._rows = rows
        self._columns = columns
        self._initUI(rows, columns)

    def _initUI(self, rows:int, columns:int):
        grid = QGridLayout()
        def slot(cv: QMaximizationConfigView):
            self.selectIndex(self._configView.index(cv))
        for row, column in np.ndindex(rows, columns):
            #box = QGroupBox(f"({row},{column})")
            box = QGroupBox()
            layout = QVBoxLayout()
            imageView = QImageView()
            configView = QMaximizationConfigView()
            configView.clicked.connect(slot)
            layout.addWidget(imageView)
            layout.addWidget(configView)
            box.setLayout(layout)
            grid.addWidget(box, row, column)
            self._boxes.append(box)
            self._imageView.append(imageView)
            self._configView.append(configView)
        self.setLayout(grid)
        self.selectIndex(0)

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by "Tab" key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)

    def connectToConfigViews(self, slot):
        import threading
        for cv in self._configView:
            # FIXME[hack]: Qt::DirectConnection: The slot is invoked immediately, when the signal is emitted.
            # This is necessary, as this method is called when setting
            # the controller, which takes place from within another Thread
            # Check details!
            cv.selected.connect(slot, Qt.DirectConnection)

    def showImage(self, image: np.ndarray, engine: Engine):
        self._imageView[self._index].setImage(image)
        self._configView[self._index].setFromEngine(engine)
        self.selectIndex((self._index + 1) % len(self._imageView))

    def selectIndex(self, index):
        if self._index is not None:
            self._boxes[self._index].setStyleSheet('')
        self._index = index
        self._boxes[self._index].setStyleSheet('QGroupBox { border: 2px solid aqua }')
        
    def keyPressEvent(self, event):
        '''Process special keys for this widgets.
        Allow moving selected entry using the cursor key.
        Allow to clear the current entry.

        Parameters
        ----------
        event : QKeyEvent
        '''
        key = event.key()
        # Space will display the current 
        if key == Qt.Key_Space:
            self._configView[self._index].select()
        elif key == Qt.Key_Delete:
            self.showImage(None, None)
        # Arrow keyes will move the selected entry
        elif key == Qt.Key_Left:
            self.selectIndex((self._index-1) % len(self._imageView))
        elif key == Qt.Key_Up:
            self.selectIndex((self._index-self._columns) % len(self._imageView))
        elif key == Qt.Key_Right:
            self.selectIndex((self._index+1) % len(self._imageView))
        elif key == Qt.Key_Down:
            self.selectIndex((self._index+self._columns) % len(self._imageView))
        else:
            event.ignore()
