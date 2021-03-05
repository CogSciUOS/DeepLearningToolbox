"""
File: logging.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

from toolbox import Toolbox
from tools.train import Training, TrainingController
from network import Network

from .panel import Panel
from qtgui.utils import QObserver
from qtgui.widgets.matplotlib import QMatplotlib
from qtgui.widgets.training import QTrainingBox

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QPushButton, QSpinBox, QLineEdit,
                             QVBoxLayout, QHBoxLayout)


class AutoencoderPanel(Panel, QObserver, qobservables={
        Network: Network.Change.all(),
        Training: Training.Change.all()}):
        # FIXME[hack]: check what we are really interested in ...
    """A panel displaying autoencoders.

    Attributes
    ----------
    _autoencoder: AutoencoderController
        A controller for a network trained as autoencoder.
    """
    _toolbox: Toolbox = None
    #_autoencoder: AutoencoderController = None

    def __init__(self, toolbox: Toolbox,
                 # autoencoderController: AutoencoderController,
                 trainingController: TrainingController,
                 parent=None) -> None:
        """Initialization of the LoggingPael.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)
        self._cache = {}

        self._initUI()
        self._layoutComponents()

        self.setToolbox(toolbox)
        self.setAutoencoderController(autoencoderController)
        self.setTrainingController(trainingController)


    # FIXME[hack]: we need a better data concept ...
    @property
    def inputs(self):
        return (None if self._toolbox is None else
                self._toolbox.get_inputs(flat=True, dtype=np.float,
                                                   test=True))

    @property
    def labels(self):
        return (None if self._toolbox is None else
                self._toolbox.labels)
    
    @property
    def imageShape(self):
        return self._toolbox.get_data_shape()

    def _initUI(self):
        """Initialize the UI elements. This will create the
        QWidgets of this :py:class:`AutoencoderPanel`, but it
        will not arange them. This will be done by
        :py:meth:`_layoutComponents`.

        """
        #
        # Controls
        #
        def slot(checked: bool):
            # FIXME[old]
            #self._autoencoder(self._toolbox.hack_new_model())
            print("FIXME[todo]: provide a new VAE Model")
        self._buttonCreateModel = QPushButton("Create")
        self._buttonCreateModel.clicked.connect(slot)

        self._editWeightsFilename = QLineEdit('vae_mlp_mnist.h5')

        def slot(checked: bool):
            self._autoencoder.load_model(self._editWeightsFilename.text())
        self._buttonLoadModel = QPushButton("Load")
        self._buttonLoadModel.clicked.connect(slot)
        
        def slot(checked: bool):
            self._autoencoder.save_model(self._editWeightsFilename.text())
        self._buttonSaveModel = QPushButton("Save")
        self._buttonSaveModel.clicked.connect(slot)
       
        def slot(checked: bool):
            self._autoencoder.plot_model()
        self._buttonPlotModel = QPushButton("Plot Model")
        self._buttonPlotModel.clicked.connect(slot)
        
        self._buttonPlotCodeDistribution = QPushButton("Code Distribution")
        self._buttonPlotCodeDistribution.clicked.\
            connect(self._onPlotCodeDistribution)
        
        self._buttonPlotCodeVisualization = QPushButton("Code Visualization")
        self._buttonPlotCodeVisualization.clicked.\
            connect(self._onPlotCodeVisualization)
        
        self._buttonPlotReconstruction = QPushButton("Plot Reconstruction")
        self._buttonPlotReconstruction.clicked.\
            connect(self._onPlotReconstruction)

        self._spinboxGridSize = QSpinBox()
        self._spinboxGridSize.setValue(10)
        self._spinboxGridSize.setRange(4,50)
        self._spinboxGridSize.valueChanged.\
            connect(self._onPlotCodeVisualization)

        #
        # Plots
        #
        self._trainingBox = QTrainingBox()
        self._pltIn = QMatplotlib()
        self._pltCode = QMatplotlib()
        self._pltOut = QMatplotlib()

    def _layoutComponents(self):
        """Layout the graphical components of this
        :py:class:`AutoencoderPanel`.
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
        buttonBar.addWidget(self._editWeightsFilename)
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

    def _enableComponents(self):
        # The "Create Model" button can be run as soon as we have an
        # Controller that is not busy
        enabled = (self._toolbox is not None and
                   self._autoencoder is not None)
        self._buttonCreateModel.setEnabled(enabled)

        # For all other buttons we also need a network
        enabled = (self._autoencoder is not None and
                   self._autoencoder.network is not None)
        for w in (self._buttonLoadModel, self._buttonSaveModel,
                  self._buttonPlotModel,
                  self._buttonPlotCodeDistribution,
                  self._spinboxGridSize, self._buttonPlotCodeVisualization,
                  self._buttonPlotReconstruction):
            w.setEnabled(enabled)

    def setToolbox(self, toolbox: Toolbox):
        self._toolbox = toolbox

    def setAutoencoderController(self, autoencoder: AutoencoderController):
        self._exchangeView('_autoencoder', autoencoder)
        self._enableComponents()
        self._trainingBox.setNetwork(autoencoder)

    def setTrainingController(self, training: TrainingController):
        self._trainingBox.setTraining(training)

    def _onPlotCodeDistribution(self, codes=None):
        """Display a 2D plot of the digit classes in the latent space.

        Arguments
        ---------
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
        """
        if isinstance(codes, np.ndarray):
            self._cache['codes'] = codes
        else:
            codes = self._cache.get('codes', None)
            
        labels = self.labels

        if codes is None:
            inputs = self.inputs
            if inputs is not None:
                self._autoencoder.\
                    encode(inputs, async_callback=self._onPlotCodeDistribution)

            self._pltCode.noData()
        else:
            with self._pltCode as ax:
                if labels is None:
                    ax.scatter(codes[:, 0], codes[:, 1])
                else:
                    ax.scatter(codes[:, 0], codes[:, 1], c=labels)
                    # plt.colorbar()
                ax.set_xlabel("z[0]")
                ax.set_ylabel("z[1]")

    def _onPlotCodeVisualization(self, images=None):
        """Plot visualization of the code space.

        Create a regular grid in code space and decode the code points
        on this grid. Construct an image showing the decoded images
        arranged on that grid.
        """

        if (isinstance(images, np.ndarray) and
            'visualization_n' in self._cache):
            # we have computed new images: redraw the figure
            n = self._cache['visualization_n']
            shape = self.imageShape
            figure = np.zeros((shape[0] * n, shape[1] * n))
            for i, (x, y) in enumerate(np.ndindex(n, n)):
                figure[y * shape[0]: (y+1) * shape[0],
                       x * shape[1]: (x+1) * shape[1]] = \
                       images[i].reshape(shape)
        elif (not isinstance(images, int) and  # triggered by _spinboxGridSize
              'visualization_figure' in self._cache and
              'visualization_n' in self._cache):
            # we have cached the figure
            n = self._cache['visualization_n']
            shape = self.imageShape
            figure = self._cache['visualization_figure']
        else:
            # we have to (re)compute the figure:
            n = self._spinboxGridSize.value()
            figure = None

        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)

        if figure is None:
            # linearly spaced coordinates corresponding to the 2D plot
            # of digit classes in the latent space
            meshgrid = np.meshgrid(grid_x, grid_y)
            grid = np.asarray([meshgrid[0].flatten(), meshgrid[1].flatten()]).T
            self._cache['visualization_n'] = n
            self._cache.pop('visualization_image', None)
            self._autoencoder.\
                decode(grid, async_callback=self._onPlotCodeVisualization)
            self._pltCode.noData()
        else:
            start_range = shape[0] // 2
            end_range = n * shape[0] - start_range + 1
            pixel_range = np.arange(start_range, end_range, shape[0])
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

    def _onPlotReconstruction(self, data=None):
        """Plot examples of the reconstructions done by the autoencoder.  This
        will display the input image next to the reconstruction as
        well as a difference image.
        """
        inputs = self.inputs
        labels = self.labels

        if isinstance(data, bool):  # invoced from GUI: select new index
            self._cache['reconstruction_index'] = \
                -1 if inputs is None else np.random.randint(len(inputs))
        elif isinstance(data, np.ndarray):
            self._cache['reconstruction_data'] = data

        index = self._cache.get('reconstruction_index', -1)
        reconstructions = self._cache.get('reconstruction_data', None)
        if reconstructions is None and inputs is not None:
            self._autoencoder.\
                reconstruct(inputs, async_callback=self._onPlotReconstruction)

        if index == -1:
            plt.noData()
        else:
            input_image = inputs[index].reshape(self.imageShape)
            input_label = None if labels is None else labels[index]
            with self._pltIn as ax:
                ax.imshow(input_image, cmap='gray')
                ax.set_title(f"input: test sample {index}" +
                             ("" if input_label is None else
                              f" ('{input_label}')"))

        outputs = reconstructions
        if outputs is None or index ==-1:
            self._pltOut.noData()
            self._pltCode.noData()
        else:
            output_image = outputs[index].reshape(self.imageShape)
            with self._pltOut as ax:
                ax.imshow(output_image, cmap='gray')
                ax.set_title("Reconstruction")

            with self._pltCode as ax:
                ax.imshow((input_image-output_image), cmap='seismic')
                ax.set_title("Differences")

    def network_changed(self, network, change):
        print(f"AutoencoderPanel.network_changed({network}, {change})")
        if 'busy_changed' in change:
            self._enableComponents()

        if 'network_changed' in change:
            self._enableComponents()
            self._cache = {}
            self._pltIn.noData()
            self._pltOut.noData()
            self._pltCode.noData()

        if 'weights_changed' in change:
            self._cache = {}

    # FIXME[hack]:
    def trainingChanged(self, training, change):
        if 'training_changed' in change:
            self._enableComponents()
            self._cache = {}

        # FIXME[todo]: it would be nice to inspect reconstruction change
        # during training
