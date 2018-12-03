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


"""
This module offers a hub to select all important constants used in the
AlexNet activation maximization module.

File: maximization.py
Author: Antonia Hain, Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""
from controller import (BaseController,
                        ActivationsController, MaximizationController)
from model import Model, ModelObserver, ModelChange
from qtgui.utils import QImageView
from qtgui.widgets import QNetworkSelector


from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics, QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import (QWidget, QLabel, QCheckBox, QLineEdit,
                             QComboBox, QPushButton,
                             QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QSizePolicy)

import cv2


from tools.am import Config, ConfigObserver, ConfigChange, EngineObserver

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
        self._config = None
        self._initUI()
        self._layoutComponents()

    def _initUI(self):
        """Add additional UI elements

            * The ``QActivationView`` showing the unit activations on the left

        """

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

        # select a random unit from the chosen layer to perform activation
        # maximization
        self._buttonRandomUnit = QPushButton("Random unit")
        self._buttonRandomUnit.setToolTip("Choose a random unit "
                                          "for visualization")

        # _unitIndex: A text field to manually enter the index of
        # desired input (UNIT_INDEX).
        self._unitIndex = QLineEdit()
        self._unitIndex.setMaxLength(8)
        self._unitIndex.setAlignment(Qt.AlignRight)
        self._unitIndex.setValidator(QIntValidator())
        self._unitIndex.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Maximum)
        self._unitIndex.setMinimumWidth(
            QFontMetrics(self.font()).width('8') * 4)
        self._unitIndex.setToolTip("Index of unit to visualize")

        self._unitMax = QLabel()
        self._unitMax.setMinimumWidth(
            QFontMetrics(self.font()).width('8') * 4)
        self._unitMax.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Expanding)

        #
        # Maximization parameters
        #

        # learning rate
        self._eta = QLineEdit()
        self._eta.setMaxLength(8)
        self._eta.setAlignment(Qt.AlignRight)
        self._eta.setValidator(QIntValidator())
        self._eta.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Maximum)
        self._eta.setMinimumWidth(
            QFontMetrics(self.font()).width('8') * 4)

        # whether maximization is initialized with random input or flat
        # colored image (RANDOMIZE_INPUT)
        self._checkRandomizeInput = QCheckBox("Initialize Input Random")

        #
        # Gaussian blur parameters
        #

        # BLUR_ACTIVATED = True
        self._checkBlur = QGroupBox("Blur")
        self._checkBlur.setCheckable(True)

        # BLUR_KERNEL = (3, 3)
        self._blurKernelWidth = QLineEdit()
        self._blurKernelHeight = QLineEdit()
        for w in self._blurKernelWidth, self._blurKernelHeight:
            w.setMaxLength(2)
            w.setAlignment(Qt.AlignRight)
            w.setValidator(QIntValidator())

        # BLUR_SIGMA = 0
        self._blurKernelSigma = QLineEdit()

        # BLUR_FREQUENCY = 5  # how many steps between two blurs. paper used 4.
        self._blurKernelFrequency = QLineEdit()

        #
        # l2 decay parameters
        #

        # L2_ACTIVATED = True
        self._checkL2Activated = QGroupBox("L2 regularization")
        self._checkL2Activated.setCheckable(True)

        # L2_LAMBDA = 0.000001  # totally arbitrarily chosen
        self._l2Lambda = QLineEdit()
        self._l2Lambda.setMaxLength(10)
        self._l2Lambda.setAlignment(Qt.AlignRight)
        self._l2Lambda.setValidator(QDoubleValidator())

        #
        # low norm pixel clipping parameters
        #

        # NORM_CLIPPING_ACTIVATED = False
        self._checkNormClipping = QGroupBox("Norm Clipping")
        self._checkNormClipping.setCheckable(True)

        # NORM_CLIPPING_FREQUENCY: how many steps between pixel clippings
        self._normClippingFrequency = QLineEdit()
        self._normClippingFrequency.setMaxLength(10)
        self._normClippingFrequency.setAlignment(Qt.AlignRight)
        self._normClippingFrequency.setValidator(QIntValidator())

        # NORM_PERCENTILE = 30  # how many of the pixels are clipped
        self._normPercentile = QLineEdit()
        self._normPercentile.setMaxLength(10)
        self._normPercentile.setAlignment(Qt.AlignRight)
        self._normPercentile.setValidator(QIntValidator())

        # low contribution pixel clipping parameters
        # see norm clipping for explanation
        # CONTRIBUTION_CLIPPING_ACTIVATED = True
        self._checkContributionClipping = QGroupBox("Contribution Clipping")
        self._checkContributionClipping.setCheckable(True)

        # CONTRIBUTION_CLIPPING_FREQUENCY = 50
        self._contributionClippingFrequency = QLineEdit()
        self._contributionClippingFrequency.setMaxLength(10)
        self._contributionClippingFrequency.setAlignment(Qt.AlignRight)
        self._contributionClippingFrequency.setValidator(QIntValidator())
        # CONTRIBUTION_PERCENTILE = 15
        self._contributionPercentile = QLineEdit()
        self._contributionPercentile.setMaxLength(10)
        self._contributionPercentile.setAlignment(Qt.AlignRight)
        self._contributionPercentile.setValidator(QIntValidator())

        # border regularizer - punishes pixel values the higher their
        # distance to the image center
        # BORDER_REG_ACTIVATED = True
        self._checkBorderReg = QGroupBox("Border Regularizer")
        self._checkBorderReg.setCheckable(True)
        # instead of summing up the product of the actual pixel to
        # center distance and its value (too strong), the effect is
        # changed by multiplying each of the resultant values with
        # this factor
        # BORDER_FACTOR = 0.000003
        self._borderFactor = QLineEdit()
        self._borderFactor.setMaxLength(10)
        self._borderFactor.setAlignment(Qt.AlignRight)
        self._borderFactor.setValidator(QDoubleValidator())

        self._borderExp = QLineEdit()
        self._borderExp.setMaxLength(10)
        self._borderExp.setAlignment(Qt.AlignRight)
        self._borderExp.setValidator(QDoubleValidator())

        self._largerImage = QGroupBox("Larger Image")
        self._largerImage.setCheckable(True)

        self._imageSizeX = QLineEdit()
        self._imageSizeX.setMaxLength(4)
        self._imageSizeX.setAlignment(Qt.AlignRight)
        self._imageSizeX.setValidator(QIntValidator())
        
        self._imageSizeY = QLineEdit()
        self._imageSizeY.setMaxLength(4)
        self._imageSizeY.setAlignment(Qt.AlignRight)
        self._imageSizeY.setValidator(QIntValidator())

        self._jitter = QGroupBox("Jitter")
        self._jitter.setCheckable(True)

        self._jitterStrength = QLineEdit()
        self._jitterStrength.setMaxLength(4)
        self._jitterStrength.setAlignment(Qt.AlignRight)
        self._jitterStrength.setValidator(QIntValidator())

        self._checkWrapAround = QCheckBox("Wrap around")
        self._checkWrapAround.setEnabled(False) # FIXME[todo]: disabling WrapAround not implemented yet

        # convergence parameters
        # relative(!) difference between loss and last 50 losses to converge to
        # LOSS_GOAL = 0.01
        self._lossGoal = QLineEdit()
        self._lossGoal.setMaxLength(5)
        self._lossGoal.setAlignment(Qt.AlignRight)
        self._lossGoal.setValidator(QDoubleValidator())

        self._lossCount = QLineEdit()
        self._lossCount.setMaxLength(5)
        self._lossCount.setAlignment(Qt.AlignRight)
        self._lossCount.setValidator(QIntValidator())


        # MAX_STEPS = 2000
        # how many steps to maximally take when optimizing an image
        self._maxSteps = QLineEdit()
        self._maxSteps.setMaxLength(5)
        self._maxSteps.setAlignment(Qt.AlignRight)
        self._maxSteps.setValidator(QIntValidator())

        # to easily switch on and off the logging of the image and loss
        # TENSORBOARD_ACTIVATED = False
        self._checkTensorboard = QCheckBox("Tensorboard output")

        # whether to save output image normalized
        # NORMALIZE_OUTPUT = True
        self._checkNormalizeOutput = QCheckBox("Normalize Output")

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
        box2.addWidget(self._checkBlur)
        box2.addWidget(self._checkL2Activated)
        box2.addStretch()

        box3 = QVBoxLayout()
        box3.addWidget(self._checkNormClipping)
        box3.addWidget(self._checkContributionClipping)
        box3.addWidget(self._checkBorderReg)
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

        # Some general remarks on QLineEdit:
        #   - textChanged(text) is emited whenever the contents of the
        #     widget changes
        #   - textEdited(text) is emited only when the user changes the
        #     text using mouse and keyboard (so it is not emitted when
        #     QLineEdit::setText() is called).
        #   - editingFinished() only is emmitted when inputMask and
        #     Validator are ok
        
        if self._config is not None:
            self._config.remove_observer(self)

        self._config = config
        if self._config is not None:
            self._config.addObserver(self)
        
        def slot(text): config.LAYER_KEY = text
        self._layerSelector.activated[str].connect(slot)

        def slot(text):
            try:
                config.UNIT_INDEX = int(text)
            except ValueError:
                config.UNIT_INDEX = 0
        self._unitIndex.textEdited.connect(slot)

        def slot(): config.random_unit()
        self._buttonRandomUnit.clicked.connect(slot)

        #
        def slot(text):
            try:
                config.ETA = int(text)
            except ValueError:
                config.ETA = 0
        self._eta.textEdited.connect(slot)

        #
        def slot(state): config.RANDOMIZE_INPUT = bool(state)
        self._checkRandomizeInput.stateChanged.connect(slot)

        #
        def slot(state): config.BLUR_ACTIVATED = bool(state)
        self._checkBlur.toggled.connect(slot)

        def slot(text):
            try:
                w = self._blurKernelWidth.text()
            except ValueError:
                w = 0
            try:
                h = self._blurKernelHeight.text()
            except ValueError:
                h = 0
            config.BLUR_KERNEL = (w,h)
        self._blurKernelWidth.textEdited.connect(slot)
        self._blurKernelHeight.textEdited.connect(slot)

        def slot(text):
            try:
                config.BLUR_SIGMA = int(text)
            except ValueError:
                config.BLUR_SIGMA = 0
        self._blurKernelSigma.textEdited.connect(slot)

        def slot(text):
            try:
                config.BLUR_FREQUENCY = int(text)
            except ValueError:
                config.BLUR_FREQUENCY = 0
        self._blurKernelFrequency.textEdited.connect(slot)

        #
        def slot(state): config.L2_ACTIVATED = bool(state)
        self._checkL2Activated.toggled.connect(slot)

        def slot(text): config.L2_LAMBDA = float(text)
        self._l2Lambda.textEdited.connect(slot)

        #
        def slot(state): config.NORM_CLIPPING_ACTIVATED = bool(state)
        self._checkNormClipping.toggled.connect(slot)

        def slot(text): config.NORM_CLIPPING_FREQUENCY = int(text)
        self._normClippingFrequency.textEdited.connect(slot)

        def slot(text): config.NORM_PERCENTILE = int(text)
        self._normPercentile.textEdited.connect(slot)

        #
        # contribution clipping
        #
        def slot(state): config.CONTRIBUTION_CLIPPING_ACTIVATED = bool(state)
        self._checkContributionClipping.toggled.connect(slot)

        def slot(text): config.CONTRIBUTION_CLIPPING_FREQUENCY = int(text)
        self._contributionClippingFrequency.textEdited.connect(slot)

        def slot(text): config.CONTRIBUTION_PERCENTILE = int(text)
        self._contributionPercentile.textEdited.connect(slot)

        #
        # border regularization
        #
        def slot(state): config.BORDER_REG_ACTIVATED = bool(state)
        self._checkBorderReg.toggled.connect(slot)

        def slot(text): config.BORDER_FACTOR = float(text)
        self._borderFactor.textEdited.connect(slot)

        def slot(text): config.BORDER_EXP = float(text)
        self._borderExp.textEdited.connect(slot)

        #
        # larger image
        #
        def slot(state): config.LARGER_IMAGE = bool(state)
        self._largerImage.toggled.connect(slot)

        def slot(text):
            try:
                w = self._imageSizeX.text()
            except ValueError:
                w = 0
            try:
                h = self._imageSizeY.text()
            except ValueError:
                h = 0
            config._IMAGE_SIZE_X = w
            config._IMAGE_SIZE_H = h
        self._blurKernelWidth.textEdited.connect(slot)
        self._blurKernelHeight.textEdited.connect(slot)

        #
        # jitter
        #
        def slot(state): config.JITTER = bool(state)
        self._jitter.toggled.connect(slot)

        def slot(text): config.JITTER_STRENGTH = int(text)
        self._jitterStrength.textEdited.connect(slot)

        #
        # wrap around
        #
        def slot(state): config.WRAP_AROUND = bool(state)
        self._checkWrapAround.stateChanged.connect(slot)

        #
        def slot(text): config.LOSS_GOAL = float(text)
        self._lossGoal.textEdited.connect(slot)

        def slot(text): config.LOSS_COUNT = int(text)
        self._lossCount.textEdited.connect(slot)

        def slot(text): config.MAX_STEPS = int(text)
        self._maxSteps.textEdited.connect(slot)

        #
        def slot(state): config.TENSORBOARD_ACTIVATED = bool(state)
        self._checkTensorboard.stateChanged.connect(slot)

        def slot(state): config.NORMALIZE_OUTPUT = bool(state)
        self._checkNormalizeOutput.stateChanged.connect(slot)

        self._updateFromConfig(config)

    def _maxUnit(self, layer_id = None) -> int:

        if layer_id is None:
            layer_id = self._config.LAYER_KEY
        if layer_id is None:
            return None

        network = self._networkSelector.network
        if network is None:
            return None
        # value = network.get_layer_input_units(layer_id)
        value = network.get_layer_output_units(layer_id)
        return value

    def _updateFromConfig(self, config: Config) -> None:
        """Update the values displayed in this QMaximizationConfig from
        the given Config object.

        Parameters
        ----------
        config: Config
        """

        #
        # update the _layerSelector
        #
        index = self._layerSelector.findText(config.LAYER_KEY)
        if index != self._layerSelector.currentIndex:
            if index != -1:
                self._layerSelector.setCurrentText(config.LAYER_KEY)
                maxUnit = self._maxUnit(config.LAYER_KEY)
                if maxUnit is None:
                    self._unitMax.setText("*")
                else:
                    self._unitMax.setText(str(maxUnit))
            else:
                self._unitMax.setText("*")


        # manually chosen unit to run activation maximization on
        self._unitIndex.setText(str(config.UNIT_INDEX))

        # learning rate
        self._eta.setText(str(config.ETA))

        # whether maximization is initialized with random input or flat
        # colored image
        self._checkRandomizeInput.setChecked(config.RANDOMIZE_INPUT)

        # gaussian blur parameters
        self._checkBlur.setChecked(config.BLUR_ACTIVATED)
        self._blurKernelWidth.setText(str(config.BLUR_KERNEL[0]))
        self._blurKernelHeight.setText(str(config.BLUR_KERNEL[1]))
        self._blurKernelSigma.setText(str(config.BLUR_SIGMA))
        self._blurKernelFrequency.setText(str(config.BLUR_FREQUENCY))

        # l2 decay parameters
        self._checkL2Activated.setChecked(config.L2_ACTIVATED)
        self._l2Lambda.setText(str(config.L2_LAMBDA))

        # low norm pixel clipping parameters
        self._checkNormClipping.setChecked(config.NORM_CLIPPING_ACTIVATED)
        self._normClippingFrequency.\
            setText(str(config.NORM_CLIPPING_FREQUENCY))
        self._normPercentile.setText(str(config.NORM_PERCENTILE))

        # low contribution pixel clipping parameters
        # see norm clipping for explanation
        self._checkContributionClipping. \
            setChecked(config.CONTRIBUTION_CLIPPING_ACTIVATED)
        self._contributionClippingFrequency. \
            setText(str(config.CONTRIBUTION_CLIPPING_FREQUENCY))
        self._contributionPercentile. \
            setText(str(config.CONTRIBUTION_PERCENTILE))

        # border regularizer - punishes pixel values the higher their
        # distance to the image center
        self._checkBorderReg.setChecked(config.BORDER_REG_ACTIVATED)
        self._borderFactor.setText(str(config.BORDER_FACTOR))
        self._borderExp.setText(str(config.BORDER_EXP))

        # larger image
        self._largerImage.setChecked(config.LARGER_IMAGE)
        self._imageSizeX.setText(str(config.IMAGE_SIZE_X))
        self._imageSizeY.setText(str(config.IMAGE_SIZE_Y))

        # jitter
        self._jitter.setChecked(config.JITTER)
        self._jitterStrength.setText(str(config.JITTER_STRENGTH))

        # wrap around
        self._checkWrapAround.setChecked(config.WRAP_AROUND)

        # convergence parameters relative(!) difference between loss
        # and last 50 losses to converge to
        self._lossGoal.setText(str(config.LOSS_GOAL))
        self._lossCount.setText(str(config.LOSS_COUNT))
        self._maxSteps.setText(str(config.MAX_STEPS))

        # to easily switch on and off the logging of the image and loss
        self._checkTensorboard.setChecked(config.TENSORBOARD_ACTIVATED)

        # whether to save output image normalized
        self._checkNormalizeOutput.setChecked(config.NORMALIZE_OUTPUT)

        self.update()

    def setController(self, controller: BaseController):
        print(f"!!! QMaximizationConfig: set Controller of type {type(controller)}")
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
                self._layerSelector.setCurrentIndex(self._layerSelector.count()-1)

    def configChanged(self, config: Config, info: ConfigChange) -> None:
        self._updateFromConfig(config)


class QMaximizationConfigView(QWidget):

    def __init__(self, parent=None, config:Config=None):
        super().__init__(parent)
        self._label = QLabel()
        self._config = None

        self.config = config

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

        self._config = config
        if self._config is not None:
            #self._config.addObserver(self)
            string = ("Layer: <b>" + str(config.LAYER_KEY) + "</b>, " +
                      "unit: <b>" + str(config.UNIT_INDEX) + "</b>")
        else:
            string = None
        self._label.setText(string)


from PyQt5.QtWidgets import QPlainTextEdit

class MyLogWindow(QPlainTextEdit):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        
    def appendMessage(self, text: str):
        self.appendPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


from PyQt5.QtWidgets import QSlider

import os

from tools.am import Engine, EngineObserver, EngineChange
from controller import MaximizationController

import numpy as np
import tensorflow as tf

from util import async
import threading


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
        self._info = QLabel("FIXME: info")
        self._button_run = QPushButton("run")
        self._button_run.clicked.connect(self.onMaximize)
        self._button_stop = QPushButton("stop")
        self._button_show = QPushButton("show")
        self._button_show.clicked.connect(self.onShow)

        self._checkbox_record = QCheckBox("Record images")
        self._info_record = QLabel()
        self._button_record_save = QPushButton("Save")
        self._button_record_save.clicked.connect(self.onSaveMovie)
        self._button_record_save.setEnabled(False)

        self._imageView = QImageView()

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

        self._logWindow = MyLogWindow()
        
        self._layoutComponents()

    def _layoutComponents(self):
        info = QVBoxLayout()
        buttons = QHBoxLayout()
        buttons.addWidget(self._button_run)
        buttons.addWidget(self._button_stop)
        buttons.addWidget(self._button_show)
        info.addLayout(buttons)
        info.addWidget(self._info)
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
        

        layout = QHBoxLayout()
        layout.addLayout(info)
        layout.addWidget(self._imageView)
        layout.addWidget(self._logWindow)
        layout.addStretch()
        
        self.setLayout(layout)

    @property
    def engine(self) -> Engine:
        return self._engine

    @engine.setter
    def engine(self, engine: Engine) -> None:
        self.logMessage(f"!!! QMaximizationControls.set_engine: ({type(engine)}) !!!")
        self._engine = engine
        self._engine.logger = self.logMessage
        self._engine.logger2 = self.logOptimizationStep

    @property
    def config(self) -> Config:
        return self._config

    @config.setter
    def config(self, config: Config) -> None:
        self.logMessage(f"!!! QMaximizationControls.set_config: ({type(config)}) !!!")
        self._config = config

    def setController(self, controller: BaseController):
        self.logMessage(f"!!! QMaximizationControls: set Controller of type {type(controller)}")
        if isinstance(controller, ActivationsController):
            super().setController(controller)
        elif isinstance(controller, MaximizationController):
            super().setController(controller, '_maximization_controller')
            self.engine = controller.get_engine()
            self.config = self._engine.config
            self._button_stop.clicked.connect(controller.onStop)


    def modelChanged(self, model: ModelObserver, info: ModelChange) -> None:

        network = model.network

        self.logMessage(f"{self.__class__.__name__}.modelChanged(): {network}")

        if info.network_changed:
            if network is not None:
                self._engine.network = network
                self._image_shape = network.get_input_shape(False)
                self.logMessage(f"input shape: {self._image_shape} ({type(self._image_shape)})")

    def engineChanged(self, engine: 'Config', info: EngineChange) -> None:

        if info.network_changed:
            self.logMessage("!!! QMaximizationControls: network changed")

        if info.config_changed:
            self.logMessage("!!! QMaximizationControls: config changed")

        if info.engine_changed:
            self.logMessage("!!! QMaximizationControls: engine changed")

        if info.image_changed:
            self.logMessage("!!! QMaximizationControls: image changed")
            self._button_record_save.setEnabled(self._stack is not None)
            if engine.image is not None:
                image_normalized = self._normalizeImage(engine.image[0])
                self._imageView.setImage(image_normalized)
                if self.display is not None:
                    self.display.showImage(image_normalized,
                                           engine._config.copy()) # FIXME[hack]: private variable


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
        if self._checkbox_record.checkState():
            if self._stack_grow:
                self._stack = np.ndarray((0,) + tuple(self._image_shape))
            else:
                self._stack = np.ndarray((self._config.MAX_STEPS+1,) +
                                         tuple(self._image_shape))
        else:
            self._stack = None
        self._stack_size = 0
        self._button_record_save.setEnabled(False)

        self._info_iteration.setText("")
        self._info_loss.setText("")
        self._info_minmax.setText("")
        self._info_mean.setText("")
        self._info_record.setText("")
        self._losses = []
        self._slider.setEnabled(self._losses is not None)

        self.logMessage("QMaximizationControls.onMaximize() -- begin")
        self._maximization_controller.onMaximize()
        self.logMessage("QMaximizationControls.onMaximize() -- end")

    def onShow(self):
        """Respond to the 'show' button.  Set the current image as
        input data to the controller.  This causes the activations
        (and the classification result) to be displayed in the
        'Activations' panel.
        """
        self._controller.onNewInput(self._imageView.getImage(),
                                    self._config.UNIT_INDEX,
                                    self._engine.description)

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
        print(f"onSaveMovie: preparing {filename} ({fourcc}) ... ")
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*fourcc),
                                 fps, frameSize, isColor=isColor)
        print(f"onSaveMovie: writer.isOpened: {writer.isOpened()}")
        if writer.isOpened():
            print(f"onSaveMovie: writing {len(self._stack_size)} frames ... ")
            # FIXME[todo]: we need the number of actual frames!
            for frame in range(self._stack_size):
                writer.write(self._normalizeImage(self._stack[frame],
                                                  as_bgr=True))
        print(f"onSaveMovie: releasing the writer")
        writer.release()
        print(f"onSaveMovie: done")

    #@async
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

        self._slider.setMaximum(len(self._losses)-1)
        # FIXME[concept]: this triggers the valueChanged signal,
        # i.e. it will call selectIteration
        self._slider.setSliderPosition(len(self._losses)-1)

        self.update()

    def _showImage(self, image: np.ndarray):
        image_min = image.min()
        image_max = image.max()
        image_mean = image.mean()
        image_std = image.std()
        self._info_minmax.setText(f"{image_min:.2f}/{image_max:.2f}"
                                  f" ({image_max-image_min:.2f})")
        self._info_mean.setText(f"{image_mean:.2f} +/- {image_std:.2f}")

        self._imageView.setImage(self._normalizeImage(image))       
        
    def logMessage(self, message: str) -> None:
        me = threading.current_thread().name
        print(f"logMessage: [{me}] {message}")
        #self._logWindow.appendMessage(message)

    def selectIteration(self, iteration: int) -> None:
        self._info_iteration.setText(f"{iteration}")
        self._info_loss.setText(f"{self._losses[iteration]:.2f}")
        if self._stack is not None:
            self._showImage(self._stack[iteration])
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
        self._index = 0
        self._boxes = []
        self._imageView = []
        self._configView = []
        self._initUI(rows, columns)

    def _initUI(self, rows:int, columns:int):
        grid = QGridLayout()
        def slot(clicked: bool): print("QGroupBox clicked {clicked}")
        for row, column in np.ndindex(rows, columns):
            #box = QGroupBox(f"({row},{column})")
            box = QGroupBox()
            layout = QVBoxLayout()
            imageView = QImageView()
            configView = QMaximizationConfigView()
            layout.addWidget(imageView)
            layout.addWidget(configView)
            box.setLayout(layout)
            grid.addWidget(box, row, column)
            self._boxes.append(box)
            self._imageView.append(imageView)
            self._configView.append(configView)
        self.setLayout(grid)
        self._boxes[0].setStyleSheet('QGroupBox { border: 2px solid aqua }')

    def showImage(self, image: np.ndarray, config: Config):
        self._imageView[self._index].setImage(image)
        self._configView[self._index].config = config
        self._boxes[self._index].setStyleSheet('')
        self._index = (self._index + 1) % len(self._imageView)
        self._boxes[self._index].setStyleSheet('QGroupBox { border: 2px solid aqua }')
