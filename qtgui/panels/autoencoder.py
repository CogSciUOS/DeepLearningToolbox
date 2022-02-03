"""
File: logging.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

# Generic imports
from typing import Optional, List, Optional
import logging

# third party imports
import numpy as np

# Qt imports
from PyQt5.QtCore import Qt, QRect, QRectF, QSize
from PyQt5.QtGui import QKeyEvent, QPaintEvent
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath
from PyQt5.QtWidgets import QWidget, QPushButton, QSpinBox, QLineEdit
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QSizePolicy
from PyQt5.QtWidgets import QLayout, QLayoutItem

# toolbox imports
from dltb.datasource import Datasource
from dltb.tool.train import Trainer
from dltb.thirdparty.tensorflow.ae import Autoencoder

from toolbox import Toolbox

# GUI imports
from .panel import Panel
from ..utils import QObserver, protect
from ..widgets.matplotlib import QMatplotlib
from ..widgets.training import QTrainingBox

# logging
LOG = logging.getLogger(__name__)


class QAutoencoderLayout(QLayout):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._items : List[QLayoutItem] = []

    def addItem(self, item: QLayoutItem) -> None:
        if len(self._items) >= 3:
            raise ValueError("Cannot add more than 3 items to "
                             "an QAutoencoderLayout")
        self._items. append(item)

    def sizeHint(self) -> QSize:
        size = QSize(100,100)
        for item in self._items:
            size.expandedTo(item.minimumSize())
        return QSize(size.width() * 3, size.height() * 2)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        width = rect.width()//3
        height = rect.height() //2
        if len(self._items) > 0:
            geom = QRect(rect.x(), rect.y(), width, height)
            self._items[0].setGeometry(geom)
        if len(self._items) > 1:
            geom = QRect(rect.x() + width, rect.y() + height//2, width, height)
            self._items[1].setGeometry(geom)
        if len(self._items) > 2:
            geom = QRect(rect.x() + 2*width, rect.y(), width, height)
            self._items[2].setGeometry(geom)

    def itemAt(self, index: int) -> Optional[QLayoutItem]:
        if not 0 <= index < self.count():
            # If there is no such item, the function must return 0.
            return None
        return self._items[index]

    def takeAt(self, index: int) -> Optional[QLayoutItem]:
        if not 0 <= index < self.count():
            # If there is no such item, the function must do nothing
            # and return 0.
            return None
        return self._items.pop(index)

    def count(self) -> None:
        return len(self._items)


class QAutoencoderWidget(QWidget):
    """A widget for displaying autoencoders.
    """

    def __init__(self, orientation: int = Qt.Horizontal, **kwargs) -> None:
        """Initialize the :py:class:`QAutoencoderWidget`.
        """
        super().__init__(**kwargs)

        # layout
        self._orientation = orientation
        self.setSizePolicy(QSizePolicy.MinimumExpanding,
                           QSizePolicy.MinimumExpanding)
        self.setStatusTip("Autoencoder")
        self.setWhatsThis("This widget displays an autoencoder .")

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by 'Tab' key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)

        self._button1 = QPushButton("B1")
        self._button2 = QPushButton("B2")
        self._button3 = QPushButton("B3")

        layout = QAutoencoderLayout()
        layout.addWidget(self._button1)
        layout.addWidget(self._button2)
        layout.addWidget(self._button3)
        self.setLayout(layout)

    def minimumSizeHint(self) -> QSize:
        """The minimum size hint.

        Returns
        -------
        QSize : The minimal size of this :py:class:`QAutoencoderWidget`
        """
        return QSize(200, 200)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Process the paint event by repainting this Widget.

        Parameters
        ----------
        event : QPaintEvent
        """
        super().paintEvent(event)  # make sure the frame is painted

        if True:  # self._autoencoder is not None
            painter = QPainter()
            painter.begin(self)
            self._drawAutoencoder(painter, event.rect())
            painter.end()

    def _drawAutoencoder(self, painter: QPainter, rect: QRect) -> None:
        """Draw a given portion of this widget.

        Parameters
        ----------
        painter:
        rect:
        """
        pen = QPen(Qt.red)
        penWidth = 1
        pen.setWidth(penWidth)
        painter.setPen(pen)

        painter.drawLine(rect.topLeft(), rect.bottomRight())

        path = QPainterPath()
        path.moveTo(rect.topLeft())
        centerRect = QRectF(rect.left() + (rect.width() *.3),
                            rect.top() + (rect.height() *.4),
                            rect.width() *.2, rect.height() *.2)
        # path.lineTo(rect.left() + (rect.width() / 2),
        #             rect.top() + (rect.height() / 2))
        path.arcTo(centerRect, 45, -90)
        path.lineTo(rect.bottomLeft())
        path.lineTo(rect.topLeft())
        path.closeSubpath()
        painter.fillPath(path, QBrush(QColor("blue")))

        path = QPainterPath()
        path.moveTo(rect.topRight())
        path.lineTo(rect.left() + (rect.width() / 2),
                    rect.top() + (rect.height() / 2))
        path.lineTo(rect.bottomRight())
        path.lineTo(rect.bottomRight())
        path.closeSubpath()
        painter.fillPath(path, QBrush(QColor("blue")))

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`QImageView` supports
        the following keys:

        space: toggle tool tips
        r: toggle the keepAspectRatio flag
        """
        key = event.key()

        if key == Qt.Key_C:  # Toggle code widget
            codeWidget = self._button2
            codeWidget.setVisible(not codeWidget.isVisible())
        else:
            super().keyPressEvent(event)


class AutoencoderPanel(Panel, QObserver, qobservables={
        Autoencoder: Autoencoder.Change.all(),
        # FIXME[hack]: check what we are really interested in ...
        Trainer: {'batch_ended', 'training_ended'}}):
    """A panel displaying autoencoders.

    The panel consists of the following parts:

    * a display for the autoencoder. The display allows to compare
      inputs and outputs of an autoencoder.  It may optionally also
      allow to visualize the codes, which is specifically interesting
      for a 2-dimensional code space.

    * controls for running a trainer, allowing to train an autoencoder.
      These controls are not specific for autoencoders.

    Attributes
    ----------
    _autoencoder: Autoencoder
        The autoencoder used by this pannel.
    """
    _toolbox: Toolbox = None
    _autoencoder: Autoencoder = None

    def __init__(self, toolbox: Optional[Toolbox] = None,
                 autoencoder: Optional[Autoencoder] = None,
                 trainer: Optional[Trainer] = None,
                 **kwargs) -> None:
        """Initialization of the LoggingPael.

        Arguments
        ---------
        toolbox:
        autoencoder:
        trainer:
        """
        super().__init__(**kwargs)
        self._cache = {}

        self._initUI()
        self._layoutComponents()

        self.setToolbox(toolbox)
        self.setAutoencoder(autoencoder)

        if trainer is None:
            trainer = Trainer()
        self.setTrainer(trainer)

    # FIXME[hack]: we need a better data concept ...
    @property
    def inputs(self):
        """The current input data (from the :py:class:`Toolbox`).
        """
        return (None if self._toolbox is None else
                self._toolbox.get_inputs(flat=True, dtype=np.float,
                                         test=True))

    @property
    def labels(self):
        """The current labels for input data (from the :py:class:`Toolbox`).
        """
        return (None if self._toolbox is None else
                self._toolbox.labels)

    @property
    def imageShape(self):
        """The shape of the current data (from the :py:class:`Toolbox`).
        """
        return (28, 28)  # FIXME[hack]
        # return self._toolbox.get_data_shape()

    def _initUI(self):
        """Initialize the UI elements. This will create the
        QWidgets of this :py:class:`AutoencoderPanel`, but it
        will not arange them. This will be done by
        :py:meth:`_layoutComponents`.

        """
        #
        # Controls
        #
        def slot(checked: bool):  # pylint: disable=unused-argument
            # FIXME[old]
            #self._autoencoder(self._toolbox.hack_new_model())
            print("FIXME[todo]: provide a new VAE Model")
        self._buttonCreateModel = QPushButton("Create")
        self._buttonCreateModel.clicked.connect(slot)

        self._editWeightsFilename = QLineEdit('vae_mlp_mnist.h5')

        # pylint: disable=function-redefined
        def slot(checked: bool):  # pylint: disable=unused-argument
            self._autoencoder.load_model(self._editWeightsFilename.text())
        self._buttonLoadModel = QPushButton("Load")
        self._buttonLoadModel.clicked.connect(slot)

        def slot(checked: bool):  # pylint: disable=unused-argument
            self._autoencoder.save_model(self._editWeightsFilename.text())
        self._buttonSaveModel = QPushButton("Save")
        self._buttonSaveModel.clicked.connect(slot)

        def slot(checked: bool):  # pylint: disable=unused-argument
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
        # FIXME[hack]:
        #
        def slot(checked: bool):  # pylint: disable=unused-argument
            data = Datasource(module='mnist')
            if self._trainer is not None:
                self._trainer.training_data = data
        self._buttonLoadMNIST = QPushButton("MNIST")
        self._buttonLoadMNIST.clicked.connect(slot)

        def slot(checked: bool):  # pylint: disable=unused-argument
            autoencoder = Autoencoder(shape=(28, 28, 1), code_dim=2)
            self.setAutoencoder(autoencoder)
        self._buttonAutoencoder = QPushButton("Autoencoder")
        self._buttonAutoencoder.clicked.connect(slot)

        self._buttonPlotRecode = QPushButton("Recode")
        self._buttonPlotRecode.clicked.\
            connect(self._onPlotReconstruction2)

        self._autoencoderWidget = QAutoencoderWidget()

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
        controlColumn = QVBoxLayout()
        controlColumn.addWidget(self._autoencoderWidget)
        
        trainingGroup = QGroupBox("Training")
        trainingGroup.setLayout(self._trainingBox.layout())
        controlColumn.addWidget(trainingGroup)
        controlColumn.addStretch()
        controlColumn.addWidget(self._buttonLoadMNIST)
        controlColumn.addWidget(self._buttonAutoencoder)
        controlColumn.addWidget(self._buttonPlotRecode)
        plotBar.addLayout(controlColumn)

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
        enabled = (self._autoencoder is not None)
        for widget in (self._buttonLoadModel, self._buttonSaveModel,
                       self._buttonPlotModel,
                       self._buttonPlotCodeDistribution,
                       self._spinboxGridSize,
                       self._buttonPlotCodeVisualization,
                       self._buttonPlotReconstruction):
            widget.setEnabled(enabled)

    def setToolbox(self, toolbox: Toolbox):
        """Set the Toolbox for this :py:class:`AutoencoderPanel`.
        """
        self._toolbox = toolbox

    def setAutoencoder(self, autoencoder: Autoencoder):
        """Set the autoencoder for this :py:class:`AutoencoderPanel`.
        """
        self._enableComponents()
        self._trainingBox.setTrainee(autoencoder)

    def setTrainer(self, trainer: Trainer) -> None:
        """Set the trainer for this :py:class:`AutoencoderPanel`.
        """
        self._trainingBox.setTrainer(trainer)

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
            with self._pltCode as axis:
                if labels is None:
                    axis.scatter(codes[:, 0], codes[:, 1])
                else:
                    axis.scatter(codes[:, 0], codes[:, 1], c=labels)
                    # plt.colorbar()
                axis.set_xlabel("z[0]")
                axis.set_ylabel("z[1]")

    def _onPlotCodeVisualization(self, images=None):
        """Plot visualization of the code space.

        Create a regular grid in code space and decode the code points
        on this grid. Construct an image showing the decoded images
        arranged on that grid.
        """

        if (isinstance(images, np.ndarray) and
            'visualization_n' in self._cache):
            # we have computed new images: redraw the figure
            number = self._cache['visualization_n']
            shape = self.imageShape
            figure = np.zeros((shape[0] * number, shape[1] * number))
            for i, (xPos, yPos) in enumerate(np.ndindex(number, number)):
                figure[yPos * shape[0]: (yPos+1) * shape[0],
                       xPos * shape[1]: (xPos+1) * shape[1]] = \
                       images[i].reshape(shape)
        elif (not isinstance(images, int) and  # triggered by _spinboxGridSize
              'visualization_figure' in self._cache and
              'visualization_n' in self._cache):
            # we have cached the figure
            number = self._cache['visualization_n']
            shape = self.imageShape
            figure = self._cache['visualization_figure']
        else:
            # we have to (re)compute the figure:
            number = self._spinboxGridSize.value()
            figure = None

        gridX = np.linspace(-4, 4, number)
        gridY = np.linspace(-4, 4, number)

        if figure is None:
            # linearly spaced coordinates corresponding to the 2D plot
            # of digit classes in the latent space
            meshgrid = np.meshgrid(gridX, gridY)
            grid = np.asarray([meshgrid[0].flatten(), meshgrid[1].flatten()]).T
            self._cache['visualization_n'] = number
            self._cache.pop('visualization_image', None)
            self._autoencoder.\
                decode(grid, async_callback=self._onPlotCodeVisualization)
            self._pltCode.noData()
        else:
            startRange = shape[0] // 2
            endRange = number * shape[0] - startRange + 1
            pixelRange = np.arange(startRange, endRange, shape[0])
            sampleRangeX = np.round(gridX, 1)
            sampleRangeY = np.round(gridY, 1)
            with self._pltCode as axis:
                axis.imshow(figure, cmap='Greys_r')
                axis.set_xticks(pixelRange, minor=False)
                axis.set_xticklabels(sampleRangeX, fontdict=None, minor=False)
                axis.set_yticks(pixelRange, minor=False)
                axis.set_yticklabels(sampleRangeY, fontdict=None, minor=False)
                axis.set_xlabel("z[0]")
                axis.set_ylabel("z[1]")
                axis.set_title("Code Layer")

    def _onPlotReconstruction(self, data=None):
        """Plot examples of the reconstructions done by the autoencoder.  This
        will display the input image next to the reconstruction as
        well as a difference image.

        """
        inputs = self.inputs
        labels = self.labels

        if isinstance(data, bool):  # invoked from GUI: select new index
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
            self._pltIn.noData()
        else:
            inputImage = inputs[index].reshape(self.imageShape)
            inputLabel = None if labels is None else labels[index]
            with self._pltIn as axis:
                axis.imshow(inputImage, cmap='gray')
                axis.set_title(f"input: test sample {index}" +
                               ("" if inputLabel is None else
                                f" ('{inputLabel}')"))

        outputs = reconstructions
        if outputs is None or index ==-1:
            self._pltOut.noData()
            self._pltCode.noData()
        else:
            outputImage = outputs[index].reshape(self.imageShape)
            with self._pltOut as axis:
                axis.imshow(outputImage, cmap='gray')
                axis.set_title("Reconstruction")

            with self._pltCode as axis:
                axis.imshow((inputImage-outputImage), cmap='seismic')
                axis.set_title("Differences")

    @protect
    def _onPlotReconstruction2(self, data=None):
        """Plot examples of the reconstructions done by the autoencoder.  This
        will display the input image next to the reconstruction as
        well as a difference image.

        This function may be invoked in two occassions: (1) to trigger
        the computation of reconstructions, or (2) to display
        reconstruction results.

        """
        # inputs, labels: input data and corresponding class labels
        inputs, labels = self._trainer.training_data[:100,('array', 'label')]

        if isinstance(data, bool):  # invoked from GUI: select new index
            self._cache['reconstruction_index'] = \
                -1 if inputs is None else np.random.randint(len(inputs))
        elif isinstance(data, np.ndarray):
            self._cache['reconstruction_data'] = data

        index = self._cache.get('reconstruction_index', -1)
        reconstructions = self._cache.get('reconstruction_data', None)
        if reconstructions is None and inputs is not None:
            # initiate a new computation
            autoencoder = self.autoencoder()
            if autoencoder is not None:
                print("Initiating another autoencoder run")
                autoencoder.recode(inputs,
                                   run_callback=self._onPlotReconstruction2)
            else:
                print("No autoencoder")

        if index == -1:
            self._pltIn.noData()
        else:
            inputImage = inputs[index].reshape(self.imageShape)
            inputLabel = None if labels is None else labels[index]
            with self._pltIn as axis:
                axis.imshow(inputImage, cmap='gray')
                axis.set_title(f"input: test sample {index}" +
                               ("" if inputLabel is None else
                                f" ('{inputLabel}')"))

        outputs = reconstructions
        if outputs is None or index ==-1:
            self._pltOut.noData()
            self._pltCode.noData()
        else:
            outputImage = outputs[index].reshape(self.imageShape)
            with self._pltOut as axis:
                axis.imshow(outputImage, cmap='gray')
                axis.set_title("Reconstruction")

            with self._pltCode as axis:
                axis.imshow((inputImage-outputImage), cmap='seismic')
                axis.set_title("Differences")

    def preparable_changed(self, autoencoder: Autoencoder, change):
        """
        """
        print(f"Preparable changed: {change}")

    def autoencoder_changed(self, autoencoder: Autoencoder, change):
        """React to changes in the autoencoder.
        """
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
    def trainer_changed(self, _trainer, change) -> None:
        """React to a change of the trainer.
        """
        print(f"AutoencoderPanel: Trainer Changed: {change}")
        if 'training_ended' in change:
            self._enableComponents()
            self._cache = {}
        if 'batch_ended' in change:
            pass  # display current training activations

        # FIXME[todo]: it would be nice to inspect reconstruction change
        # during training
