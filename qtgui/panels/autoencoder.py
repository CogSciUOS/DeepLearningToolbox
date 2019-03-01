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

from toolbox import toolbox

# FIXME[hack]
from network.keras import ObservableCallback

from models.example_keras_vae_mnist import KerasAutoencoder

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.datasets import mnist


class AutoencoderPanel(Panel, toolbox.Observer):
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
        self._progress = ObservableCallback()

        self._initDataset()

        # h5 model trained weights
        self._weights_file = 'vae_mlp_mnist.h5'

        self._initUI()
        self._layoutComponents()
        self._connectComponents()

        # Training parameters
        self._spinboxEpochs.setValue(4)
        self._spinboxBatchSize.setValue(128)

        toolbox.addObserver(self)
        self._enableComponents()

        # 
        self._trainingBox.observe(self._progress)

    def _initDataset(self):
        """Initialize the dataset.
        This will set the self._x_train, self._y_train, self._x_test, and
        self._y_test variables. Although the actual autoencoder only
        requires the x values, for visualization the y values (labels)
        may be interesting as well.

        The data will be flattened (stored in 1D arrays), converted to
        float32 and scaled to the range 0 to 1. 
        """
        #
        # The dataset
        #
        
        # load the MNIST dataset
        x, y = mnist.load_data()
        (self._x_train, self._y_train) = x
        (self._x_test, self._y_test) = y

        input_shape = self._x_train.shape[1:]
        original_dim = input_shape[0] * input_shape[1]
        self._x_train = np.reshape(self._x_train, [-1, original_dim])
        self._x_test = np.reshape(self._x_test, [-1, original_dim])
        self._x_train = self._x_train.astype('float32') / 255
        self._x_test = self._x_test.astype('float32') / 255


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
        self._buttonPlotResults = QPushButton("Plot Results")
        self._buttonPlotReconstruction = QPushButton("Plot Reconstruction")

        self._spinboxEpochs = QSpinBox()
        self._spinboxEpochs.setRange(1, 50)

        self._spinboxBatchSize = QSpinBox()
        self._spinboxBatchSize.setRange(1, 256)

        #
        # Plots
        #
        self._trainingBox = QTrainingBox()
        self._resultPlot0 = QMatplotlib()
        self._resultPlot1 = QMatplotlib()
        self._resultPlot2 = QMatplotlib()

    def _connectComponents(self):
        self._buttonCreateModel.clicked.connect(self._onCreateModel)
        self._buttonTrainModel.clicked.connect(self._onTrainModel)
        self._buttonLoadModel.clicked.connect(self._onLoadModel)
        self._buttonSaveModel.clicked.connect(self._onSaveModel)
        self._buttonPlotModel.clicked.connect(self._onPlotModel)
        self._buttonPlotResults.clicked.connect(self._onPlotResults)
        self._buttonPlotReconstruction.clicked.connect(self._onPlotReconstruction)

    def _layoutComponents(self):
        """Layout the UI elements.

            * The ``QLogHandler`` displaying the log messages

        """
        plotBar = QHBoxLayout()
        plotBar.addWidget(self._trainingBox)
        plotBar.addWidget(self._resultPlot0)
        plotBar.addWidget(self._resultPlot1)
        plotBar.addWidget(self._resultPlot2)

        buttonBar = QHBoxLayout()
        buttonBar.addWidget(self._buttonCreateModel)
        buttonBar.addWidget(self._spinboxEpochs)
        buttonBar.addWidget(self._spinboxBatchSize)
        buttonBar.addWidget(self._buttonTrainModel)
        buttonBar.addWidget(self._buttonLoadModel)
        buttonBar.addWidget(self._buttonSaveModel)
        buttonBar.addWidget(self._buttonPlotModel)
        buttonBar.addWidget(self._buttonPlotResults)
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
        self._buttonTrainModel.setEnabled(available)
        self._buttonLoadModel.setEnabled(available)
        self._buttonSaveModel.setEnabled(available)
        self._buttonPlotModel.setEnabled(available)
        self._buttonPlotResults.setEnabled(available)
        self._buttonPlotReconstruction.setEnabled(available)

    def _onLoadModel(self):
        self._autoencoder.load(self._weights_file)
        
    def _onSaveModel(self):
        self._autoencoder.save(self._weights_file)

    def _onCreateModel(self):
        # Initialize the network
        #
        # Network parameters
        #
        original_dim = self._x_train.shape[1]
        intermediate_dim = 512
        latent_dim = 2
        import util
        util.runner.runTask(self._createModel, original_dim)

    def _createModel(self, original_dim):
        self._autoencoder = KerasAutoencoder(original_dim)
        self._enableComponents()

    def _onTrainModel(self):
        import util # FIXME[hack]
        util.runner.runTask(self._trainModel)
        
    def _trainModel(self):
        epochs = self._spinboxEpochs.value()
        batchSize = self._spinboxBatchSize.value()
        self._autoencoder.train(self._x_train, self._x_test,
                                epochs=epochs,
                                batch_size=batchSize,
                                progress=self._progress)

    def _onPlotModel(self):
        pass

    def _onPlotResults(self):
        data = (self._x_test, self._y_test)
        batchSize = self._spinboxBatchSize.value()
        self.plot_results1(data, batch_size=batchSize)

        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 28
        self.plot_results2(n, digit_size)

    def _onPlotReconstruction(self):
        data = (self._x_test, self._y_test)
        batchSize = self._spinboxBatchSize.value()
        self.plotReconstruction(data, batch_size=batchSize)


    def plot_results1(self, data, batch_size=128, model_name="vae_mnist"):
        """Plots labels and MNIST digits as function of 2-dim latent vector
        
        Arguments
        ---------
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
        """
        filename = os.path.join(model_name, "vae_mean.png")

        x_test, y_test = data
        os.makedirs(model_name, exist_ok=True)

        # display a 2D plot of the digit classes in the latent space
        z_mean = self._autoencoder.encode(x_test, batch_size)
        plt = self._resultPlot1
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
        #plt.colorbar()
        #plt.xlabel("z[0]")
        #plt.ylabel("z[1]")
        #plt.savefig(filename)
        #plt.show()

    def plot_results2(self, n, digit_size, model_name="vae_mnist"):
        filename = os.path.join(model_name, "digits_over_latent.png")

        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self._autoencoder.decode(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

        start_range = digit_size // 2

        end_range = n * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)

        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)

        plt = self._resultPlot2
        #plt.xticks(pixel_range, sample_range_x)
        #plt.yticks(pixel_range, sample_range_y)
        #plt.xlabel("z[0]")
        #plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        #plt.savefig(filename)
        #plt.show()

    def plotReconstruction(self, data, batch_size=128):
        idx = np.random.randint(len(data[0]))
        reconstruction = self._autoencoder.reconstruct(data[0][idx:idx+1],
                                                       batch_size=batch_size)
        
        plt = self._resultPlot0
        plt.imshow(data[0][idx].reshape(28,28), cmap='Greys_r')
        
        plt = self._resultPlot1
        plt.imshow(reconstruction[0].reshape(28,28), cmap='Greys_r')

        plt = self._resultPlot2
        plt.imshow(data[0][idx].reshape(28,28) -
                   reconstruction[0].reshape(28,28), cmap='Greys_r')

    def toolboxChanged(self, toolbox, change):
        self._enableComponents(toolbox.locked())

