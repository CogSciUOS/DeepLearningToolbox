"""
File: logging.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""
# FIXME[hack]: this is just using a specific keras network as proof of
# concept. It has to be modularized and integrated into the framework

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QPushButton, QSpinBox,
                             QVBoxLayout, QHBoxLayout)

from .panel import Panel
from qtgui.widgets.matplotlib import QMatplotlib
from qtgui.widgets.training import QTrainingBox

import numpy as np
import matplotlib.pyplot as plt
import os

from qtgui.utils import QObserver

from toolbox import toolbox
from tools.train import Training, Trainer

# FIXME[hack]
from network.keras import Training as KerasTraining

class AutoencoderPanel(Panel, QObserver, toolbox.Observer, Training.Observer):
    """A panel displaying autoencoders.

    Attributes
    ----------
    _autoencoder: Network
        A network trained as autoencoder.

    _x_train
    _x_test

    _y_train
    _y_test

    """

    def __init__(self, parent=None):
        """Initialization of the LoggingPael.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)
        self._autoencoder = None
        self._dataAction = None
        self._training = KerasTraining()
        self._trainer = Trainer(self._training)

        self._training.hack_load_mnist()
        # FIXME[hack]
        #self._x_train = self._training._x_train
        #self._y_train = self._training._y_train
        self._x_test = self._training._x_test
        self._y_test = self._training._y_test
        self._imageShape = (28, 28)

        self._initUI()
        self._layoutComponents()
        self._connectComponents()

        controller = AutoencoderController()
        self.setAutoencoderController(controller)

        # Training parameters
        self._spinboxEpochs.setValue(4)

        self.observe(toolbox)
        self._enableComponents()

        # 
        self._trainingBox.observe(self._training)
        self.observe(self._training)


    def _initUI(self):
        """Add the UI elements

            * The ``QLogHandler`` showing the log messages

        """
        #
        # Controls
        #
        self._buttonCreateModel = QPushButton("Create")
        self._buttonTrainModel = QPushButton("Train")
        self._buttonLoadModel = QPushButton("Load")
        self._buttonSaveModel = QPushButton("Save")
        self._buttonPlotModel = QPushButton("Plot Model")
        self._buttonPlotCodeDistribution = QPushButton("Code Distribution")
        self._buttonPlotCodeVisualization = QPushButton("Code Visualization")
        self._buttonPlotReconstruction = QPushButton("Plot Reconstruction")

        self._spinboxEpochs = QSpinBox()
        self._spinboxEpochs.setRange(1, 50)

        self._spinboxBatchSize = QSpinBox()

        self._spinboxGridSize = QSpinBox()
        self._spinboxGridSize.setValue(10)
        self._spinboxGridSize.setRange(4,50)

        #
        # Plots
        #
        self._trainingBox = QTrainingBox()
        self._pltIn = QMatplotlib()
        self._pltCode = QMatplotlib()
        self._pltOut = QMatplotlib()

    def _connectComponents(self):
        self._buttonCreateModel.clicked.connect(self._trainer._hackNewModel)
        self._buttonTrainModel.clicked.connect(self._trainer.start)
        self._spinboxEpochs.valueChanged.connect(self._trainer.set_epochs)
        self._spinboxBatchSize.valueChanged.\
            connect(self._trainer.set_batch_size)
        self._buttonPlotCodeDistribution.clicked.\
            connect(self._onPlotCodeDistribution)
        self._buttonPlotCodeVisualization.clicked.\
            connect(self._onPlotCodeVisualization)
        self._buttonPlotReconstruction.clicked.\
            connect(self._onPlotReconstruction)

    def _layoutComponents(self):
        """Layout the UI elements.

            * The ``QLogHandler`` displaying the log messages

        """
        plotBar = QHBoxLayout()
        plotBar.addWidget(self._trainingBox)

        displayBox = QHBoxLayout()
        displayBox.addWidget(self._pltCode)
        displayInOut = QVBoxLayout()
        displayInOut.addWidget(self._pltIn)
        displayInOut.addWidget(self._pltOut)
        displayBox.addLayout(displayInOut)
        plotBar.addLayout(displayBox)

        buttonBar = QHBoxLayout()
        buttonBar.addWidget(self._buttonCreateModel)
        buttonBar.addWidget(self._spinboxEpochs)
        buttonBar.addWidget(self._spinboxBatchSize)
        buttonBar.addWidget(self._buttonTrainModel)
        buttonBar.addWidget(self._buttonLoadModel)
        buttonBar.addWidget(self._buttonSaveModel)
        buttonBar.addWidget(self._buttonPlotModel)
        buttonBar.addWidget(self._buttonPlotCodeDistribution)
        buttonBar.addWidget(self._spinboxGridSize)
        buttonBar.addWidget(self._buttonPlotCodeVisualization)
        buttonBar.addWidget(self._buttonPlotReconstruction)

        layout = QVBoxLayout()
        layout.addLayout(plotBar)
        layout.addLayout(buttonBar)
        self.setLayout(layout)

    def _enableComponents(self, running=False):
        available = self._autoencoder is not None and not running
        self._buttonCreateModel.setEnabled(not running)
        self._spinboxEpochs.setEnabled(available)
        self._spinboxBatchSize.setEnabled(available)
        self._buttonTrainModel.setEnabled(self._autoencoder is not None)
        for w in (self._buttonLoadModel, self._buttonSaveModel,
                  self._buttonPlotModel, 
                  self._buttonPlotCodeDistribution,
                  self._spinboxGridSize, self._buttonPlotCodeVisualization,
                  self._buttonPlotReconstruction):
            w.setEnabled(available)

    def setAutoencoderController(self, controller):
        self._autoencoderController = controller
        self._spinboxBatchSize.setRange(*controller.batch_size_range)
        self._spinboxBatchSize.valueChanged.\
            connect(AutoencoderController.batch_size.fset.__get__(controller))
        self._buttonLoadModel.clicked.connect(controller.load_model)
        self._buttonSaveModel.clicked.connect(controller.save_model)
        self._buttonPlotModel.clicked.connect(controller.plot_model)
        self.autoencoderControllerChanged(controller, controller.Change.all())
        self.observe(controller)

    def _onPlotCodeDistribution(self):
        """Plots labels and MNIST digits as function of 2-dim latent vector
        
        Arguments
        ---------
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
        """
        self._dataAction = None
        self._autoencoderController.set_input_data(self._x_test, self._y_test)
        self._dataAction = self.plotCodeDistribution
        self._autoencoderController.encode()

    def plotCodeDistribution(self):
        """Display a 2D plot of the digit classes in the latent space.

        This method make use of precomputed data stored in the
        controller: codes and labels.
        """
        z_mean = self._autoencoderController.code_means
        y_test = self._autoencoderController.labels

        plt = self._pltCode
        if z_mean is None:
            plt.noData()
        elif y_test is None:
            plt.scatter(z_mean[:, 0], z_mean[:, 1])
        else:
            plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
            # plt.colorbar()
            # plt.xlabel("z[0]")
            # plt.ylabel("z[1]")

            # filename = os.path.join(model_name, "vae_mean.png")
            # os.makedirs(model_name, exist_ok=True)
            # plt.savefig(filename)
            # plt.show()

    def _onPlotCodeVisualization(self):

        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        n = self._spinboxGridSize.value()
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)
        meshgrid = np.meshgrid(grid_x, grid_y)
        grid = np.asarray([meshgrid[0].flatten(), meshgrid[1].flatten()]).T

        batch_size = self._spinboxBatchSize.value()

        self._dataAction = None
        self._autoencoderController.set_code_data(grid)
        self._autoencoderController.set_batch_size(batch_size)
        self._dataAction = self.plotCodeVisualization
        self._autoencoderController.decode()

    def plotCodeVisualization(self):
        """Plot visualization of the code space.

        Create a regular grid in code space and decode the code points
        on this grid. Construct an image showing the decoded images
        arranged on that grid.
        """

        codes = self._autoencoderController.codes
        images = self._autoencoderController.outputs

        if images is None:
            self._pltCode.noData()
        else:
            n = int(round(np.sqrt(len(images))))
            shape = self._imageShape
            figure = np.zeros((shape[0] * n, shape[1] * n))
            for i, (x, y) in enumerate(np.ndindex(n, n)):
                figure[y * shape[0]: (y+1) * shape[0],
                       x * shape[1]: (x+1) * shape[1]] = \
                       images[i].reshape(shape)
            start_range = shape[0] // 2
            end_range = n * shape[0] - start_range + 1
            pixel_range = np.arange(start_range, end_range, shape[0])
            grid_x = np.linspace(-4, 4, n)
            grid_y = np.linspace(-4, 4, n)
            sample_range_x = np.round(grid_x, 1)
            sample_range_y = np.round(grid_y, 1)
            with self._pltCode as ax:
                ax.imshow(figure, cmap='Greys_r')
                ax.set_xticks(pixel_range, minor=False)
                ax.set_xticklabels(sample_range_x, fontdict=None, minor=False)
                ax.set_yticks(pixel_range, minor=False)
                ax.set_yticklabels(sample_range_y, fontdict=None, minor=False)
                ax.set_xlabel("z[0]")
                ax.set_ylabel("z[1]")
                ax.set_title("Code Layer")

            #FIXME[old]:
            # model_name="vae_mnist"
            # filename = os.path.join(model_name, "digits_over_latent.png")
            # plt.savefig(filename)
            # plt.show()
        
    def _onPlotReconstruction(self):
        self._dataAction = None
        self._autoencoderController.set_input_data(self._x_test, self._y_test)
        self._dataAction = self.plotReconstruction
        self._autoencoderController.reconstruct()

    def plotReconstruction(self):
        inputs = self._autoencoderController.inputs
        if inputs is None:
            idx = -1 
            plt.noData()
        else:
            idx = np.random.randint(len(inputs))
            self._pltIn.imshow(inputs[idx].reshape(self._imageShape),
                               cmap='Greys_r')

        outputs = self._autoencoderController.outputs
        if outputs is None or idx ==-1:
            self._pltOut.noData()
            self._pltCode.noData()
        else:
            plt = self._pltOut
            plt.imshow(outputs[idx].reshape(self._imageShape), cmap='Greys_r')

            plt = self._pltCode
            plt.imshow((inputs[idx]-outputs[0]).reshape(self._imageShape),
                       cmap='seismic')

    def toolboxChanged(self, toolbox, change):
        self._enableComponents(toolbox.locked())

    def trainingChanged(self, training, change):
        if 'training_changed' in change and self._trainer is not None:
            self._buttonTrainModel.clicked.disconnect()
            if training.running:
                self._buttonTrainModel.setText("Stop")
                self._buttonTrainModel.clicked.connect(self._trainer.stop)
            else:
                self._buttonTrainModel.setText("Train")
                self._buttonTrainModel.clicked.connect(self._trainer.start)

        if 'network_changed' in change:
            self._autoencoder = training.network
            self._autoencoderController.set_autoencoder(training.network)
            self._enableComponents()

    def autoencoderControllerChanged(self, controller, change):
        if 'busy_changed' in change:
            self._enableComponents(controller.busy)

        if 'data_changed' in change:
            if self._dataAction is not None:
                self._dataAction()

        if 'parameter_changed' in change:
            self._spinboxBatchSize.setValue(controller.batch_size)

def busy(f):
    def deco_helper(self):
        f(self)
        self._busy = True
        self.notifyObservers('busy_changed')
    
    def deco(self):
        print('In deco')
        if self._busy:
            raise RuntimeException("Autoencoder is currently busy")
        self._busy = True
        self.notifyObservers('busy_changed')
        self._runner.runTask(deco_helper, self)

    return deco


from base.observer import Observable
import util

class AutoencoderController(Observable, method='autoencoderControllerChanged',
                            changes=['busy_changed', 'data_changed',
                                     'parameter_changed']):
    # 'parameter_changed': hyperparmeter changed (batch_size)

    def __init__(self, autoencoder=None):
        super().__init__()
        self.set_autoencoder(autoencoder)
        self._busy = False

        self._batch_size = 128
        self._batch_size_min = 1
        self._batch_size_max = 256
        # h5 model trained weights
        self._weights_file = 'vae_mlp_mnist.h5'

    def set_autoencoder(self, autoencoder):
        self._autoencoder = autoencoder

    def load_model(self):
        util.runner.runTask(self._load_model)

    def _load_model(self):
        self._busy = True
        self.change('busy_changed')
        self._autoencoder.load(self._weights_file)
        self._busy = False
        self.change('busy_changed')
        
    def save_model(self):
        if self._busy:
            raise RuntimeException("Autoencoder is currently busy")
        util.runner.runTask(self._save_model)
        
    def _save_model(self):
        self._busy = True
        self.change('busy_changed')
        self._autoencoder.save(self._weights_file)
        self._busy = False
        self.change('busy_changed')

    def plot_model(self):
        pass

    @property
    def busy(self):
        return self._busy

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size: int):
        if self._batch_size_min <= size <= self._batch_size_max:
            if self._batch_size != size:
                self._batch_size = size
                self.change('parameter_changed')
        else:
            raise ValueError(f"Invalid batch size {size}:"
                             f"allowed range is from {self._batch_size_min}"
                             f"to {self._batch_size_max}")

    @property
    def batch_size_min(self) -> int:
        return self._batch_size_min

    @property
    def batch_size_max(self) -> int:
        return self._batch_size_max

    @property
    def batch_size_range(self):
        return (self._batch_size_min, self._batch_size_max)

    @property
    def inputs(self):
        return self._input_data

    @property
    def labels(self):
        return self._input_labels

    @property
    def code_means(self):
        return self._z_mean

    @property
    def codes(self):
        return self._code_data

    @property
    def outputs(self):
        return self._output_data

    def set_input_data(self, data, labels=None):
        self._input_data = data
        self._input_labels = labels
        self._z_mean = None
        self.set_code_data(None)

    def set_code_data(self, data, labels=None):
        self._code_data = data
        self._code_labels = labels
        self.set_output_data(None)

    def set_output_data(self, data, labels=None):
        self._output_data = data
        self._output_labels = labels

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def encode(self):
        if self._busy:
            raise RuntimeException("Autoencoder is currently busy")
        util.runner.runTask(self._encode)
        
    def _encode(self):
        self._z_mean = None
        self._busy = True
        self.change('busy_changed', 'data_changed')
        self._z_mean = self._autoencoder.encode(self._input_data,
                                                self._batch_size)
        self._busy = False
        self.change('busy_changed', 'data_changed')

    def decode(self):
        if self._busy:
            raise RuntimeException("Autoencoder is currently busy")
        util.runner.runTask(self._decode)

    def _decode(self):
        self._output_data = None
        self._output_labels = None
        self._busy = True
        self.change('busy_changed', 'data_changed')
        self._output_data = self._autoencoder.decode(self._code_data,
                                                     self._batch_size)
        self._busy = False
        self.change('busy_changed', 'data_changed')

    def reconstruct(self):
        if self._busy:
            raise RuntimeException("Autoencoder is currently busy")
        util.runner.runTask(self._reconstruct)

    def _reconstruct(self):
        self._z_mean = None
        self._busy = True
        self.change('busy_changed', 'data_changed')
        self._output_data = self._autoencoder.reconstruct(self._input_data,
                                                          self._batch_size)
        self._busy = False
        self.change('busy_changed', 'data_changed')

    def sample(self):
        if self._busy:
            raise RuntimeException("Autoencoder is currently busy")
