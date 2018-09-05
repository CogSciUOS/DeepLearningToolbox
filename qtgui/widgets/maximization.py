"""
This module offers a hub to select all important constants used in the
AlexNet activation maximization module.

File: maximization.py
Author: Antonia Hain, Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""
import observer
from controller import ActivationsController

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics, QIntValidator
from PyQt5.QtWidgets import QWidget, QLabel, QCheckBox, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox, QLineEdit, QSizePolicy


class QMaximizationConfig(QWidget, observer.Observer):


    # in which layer the unit is found. needs to be a key in layer_dict in
    # am_alexnet
    LAYER_KEY = 'fc8'

    # whether a unit from the chosen layer to perform activation
    # maximization on should be chosen at random
    RANDOM_UNIT = True

    # this is only used if RANDOM_UNIT is set to False manually chosen
    # unit to run activation maximization on
    UNIT_INDEX = 947

    # learning rate
    ETA = 500

    # whether maximization is initialized with random input or flat
    # colored image
    RANDOMIZE_INPUT = True

    # gaussian blur parameters
    BLUR_ACTIVATED = True
    BLUR_KERNEL = (3, 3)
    BLUR_SIGMA = 0
    BLUR_FREQUENCY = 5  # how many steps between two blurs. paper used 4.

    # l2 decay parameters
    L2_ACTIVATED = True
    L2_LAMBDA = 0.000001  # totally arbitrarily chosen

    # low norm pixel clipping parameters
    NORM_CLIPPING_ACTIVATED = False
    NORM_CLIPPING_FREQUENCY = 25  # how many steps between pixel clippings
    NORM_PERCENTILE = 30  # how many of the pixels are clipped

    # low contribution pixel clipping parameters
    # see norm clipping for explanation
    CONTRIBUTION_CLIPPING_ACTIVATED = True
    CONTRIBUTION_CLIPPING_FREQUENCY = 50
    CONTRIBUTION_PERCENTILE = 15

    # border regularizer - punishes pixel values the higher their distance to
    # the image center
    BORDER_REG_ACTIVATED = True
    # instead of summing up the product of the actual pixel to center distance
    # and its value (too strong), the effect is changed by multiplying each of
    # the resultant values with this factor
    BORDER_FACTOR = 0.000003

    # convergence parameters
    # relative(!) difference between loss and last 50 losses to converge to
    LOSS_GOAL = 0.01
    MAX_STEPS = 2000  # how many steps to maximally take when optimizing an image
    
    # to easily switch on and off the logging of the image and loss
    TENSORBOARD_ACTIVATED = False

    # whether to save output image normalized
    NORMALIZE_OUTPUT = True


    def __init__(self, parent=None):
        '''Initialization of the ActivationsPael.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        model   :   model.Model
                    The backing model. Communication will be handled by a
                    controller.
        '''
        super().__init__(parent)
        self._initUI()


    def _initUI(self):
        '''Add additional UI elements

            * The ``QActivationView`` showing the unit activations on the left

        '''

        ##
        ## Unit selection
        ##
        
        # in which layer the unit is found. needs to be a key in layer_dict in
        # am_alexnet
        LAYER_KEY = 'fc8'
        self._layerSelector = QComboBox()

        # whether a unit from the chosen layer to perform activation
        # maximization on should be chosen at random
        RANDOM_UNIT = True
        self._checkRandom = QCheckBox("Random unit")

        # this is only used if RANDOM_UNIT is set to False manually chosen
        # unit to run activation maximization on
        UNIT_INDEX = 947

        ## _unitIndex: A text field to manually enter the index of
        ## desired input.
        self._unitIndex = QLineEdit()
        self._unitIndex.setMaxLength(8)
        self._unitIndex.setAlignment(Qt.AlignRight)
        self._unitIndex.setValidator(QIntValidator())
        #self._unitIndex.textChanged.connect(self._editIndex)
        #self._unitIndex.textEdited.connect(self._editIndex)
        self._unitIndex.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Maximum)
        self._unitIndex.setMinimumWidth(
            QFontMetrics(self.font()).width('8') * 4)
        self._unitIndex.setText(str(self.UNIT_INDEX))


        self._unitMax = QLabel()
        self._unitMax.setMinimumWidth(
            QFontMetrics(self.font()).width('8') * 4)
        self._unitMax.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Expanding)


        ##
        ## Maximization parameters
        ##

        ## learning rate
        self._learningRate = QLineEdit()
        self._learningRate.setMaxLength(8)
        self._learningRate.setAlignment(Qt.AlignRight)
        self._learningRate.setValidator(QIntValidator())
        self._learningRate.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Maximum)
        self._learningRate.setMinimumWidth(
            QFontMetrics(self.font()).width('8') * 4)
        self._learningRate.setText(str(self.ETA))

        ## whether maximization is initialized with random input or flat
        ## colored image
        RANDOMIZE_INPUT = True
        self._initializeInputRandom = QCheckBox("Initialize Input Random")

        # gaussian blur parameters
        BLUR_ACTIVATED = True
        BLUR_KERNEL = (3, 3)
        BLUR_SIGMA = 0
        BLUR_FREQUENCY = 5  # how many steps between two blurs. paper used 4.

        # l2 decay parameters
        L2_ACTIVATED = True
        L2_LAMBDA = 0.000001  # totally arbitrarily chosen

        # low norm pixel clipping parameters
        NORM_CLIPPING_ACTIVATED = False
        NORM_CLIPPING_FREQUENCY = 25  # how many steps between pixel clippings
        NORM_PERCENTILE = 30  # how many of the pixels are clipped

        # low contribution pixel clipping parameters
        # see norm clipping for explanation
        CONTRIBUTION_CLIPPING_ACTIVATED = True
        CONTRIBUTION_CLIPPING_FREQUENCY = 50
        CONTRIBUTION_PERCENTILE = 15

        # border regularizer - punishes pixel values the higher their
        # distance to the image center
        BORDER_REG_ACTIVATED = True
        # instead of summing up the product of the actual pixel to
        # center distance and its value (too strong), the effect is
        # changed by multiplying each of the resultant values with
        # this factor
        BORDER_FACTOR = 0.000003

        # convergence parameters
        # relative(!) difference between loss and last 50 losses to converge to
        LOSS_GOAL = 0.01
        MAX_STEPS = 2000  # how many steps to maximally take when optimizing an image
    
        # to easily switch on and off the logging of the image and loss
        TENSORBOARD_ACTIVATED = False
    
        # whether to save output image normalized
        NORMALIZE_OUTPUT = True


        unitLayout = QVBoxLayout()
        unitLayout.addWidget(self._unitIndex)
        unitLayout.addWidget(self._unitMax)
        unitLayout.addWidget(self._checkRandom)
        unitLayout.addWidget(self._layerSelector)

        unitLayout.addWidget(self._learningRate)

        unitLayout.addWidget(self._initializeInputRandom)
        self.setLayout(unitLayout)
        

    def setController(self, controller : ActivationsController):
        super().setController(controller)
        self._layerSelector.currentIndexChanged[str].connect(controller.onLayerSelected)



    def modelChanged(self, model, info):

        network = model.getNetwork()
        
        if info.network_changed:
            self._layerSelector.clear()
            if network is not None:
                self._layerSelector.addItems(network.layer_dict.keys())

        if info.layer_changed:
            if network is not None:
                layer_id = model._layer
                #self._layerSelector.setCurrentIndex(layer_id)
            else:
                layer_id = None
                
            if layer_id is not None:
                self._layerSelector.setCurrentText(layer_id)
                #self._unitMax.setText(str(network.get_layer_input_units(layer_id)))
                self._unitMax.setText(str(network.get_layer_output_units(layer_id)))
            else:
                self._unitMax.setText("")
