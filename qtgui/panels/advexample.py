"""
File: logging.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

# standard imports
from base import Runner
from toolbox import Toolbox

# Qt imports
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QPushButton, QSpinBox,
                             QVBoxLayout, QHBoxLayout)

# toolbox imports
from toolbox import Toolbox
from dltb.base.data import Data
from dltb.base.image import Image, Imagelike

# GUI imports
from .panel import Panel
from ..utils import QObserver
from ..widgets.matplotlib import QMatplotlib
from ..widgets.training import QTrainingBox
from ..widgets.data import QDataInspector


class AdversarialExamplePanel(Panel, QObserver, qobservables={
        Toolbox: {'input_changed'}}):
    """A panel displaying adversarial examples.

    Attributes
    ----------
    _network: NetworkView
        A network trained as autoencoder.
    """

    def __init__(self, toolbox: Toolbox = None, **kwargs):
        """Initialization of the AdversarialExamplePanel.

        Parameters
        ----------
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(**kwargs)
        self._controller = None  # FIXME[old]

        self._initUI()
        self._layoutUI()
        self.setToolbox(toolbox)

        # FIXME[old]
        # self.setController(AdversarialExampleController())
        

    def _initUI(self):
        """Initialize the user interface.

        The user interface contains the following elements:
        * the data selector: depicting the current input image
          and allowing to select new inputs from a datasource
        * ouput: adversarial example
        * ouput: adversarial perturbation
        * ouput: statistics
        """

        #
        # Input data
        #
        self._dataInspector = QDataInspector()
        self._dataView = self._dataInspector.dataView()
        self._dataView.addAttribute('filename')
        self._dataView.addAttribute('basename')
        self._dataView.addAttribute('directory')
        self._dataView.addAttribute('path')
        self._dataView.addAttribute('regions')
        self._dataView.addAttribute('image')

        #
        # Controls
        #
        self._buttonCreateModel = QPushButton("Create")
        self._buttonTrainModel = QPushButton("Train")
        self._buttonLoadModel = QPushButton("Load")
        self._buttonSaveModel = QPushButton("Save")
        self._buttonResetModel = QPushButton("Reset")
        self._buttonPlotModel = QPushButton("Plot Model")
        self._buttonShowExample = QPushButton("Show")
        self._buttonShowExample.clicked.connect(self._onShowExample)

        #
        # Plots
        #
        self._trainingBox = QTrainingBox()
        self._pltOriginal = QMatplotlib()
        self._pltAdversarial = QMatplotlib()

    def _layoutUI(self):
        """Layout the UI elements.
        """
        # The big picture:
        #
        #  +--------------------+----------------------------------------+
        #  |+------------------+|+------------------------------------+  |
        #  ||dataInspector     ||| Result                             |  |
        #  ||[view]            ||| (Adversarial Example)              |  |
        #  ||                  |||                                    |  |
        #  ||                  |||                                    |  |
        #  ||                  ||| Diffs                              |  |
        #  ||                  ||| (Adversarial Perturbation)         |  |
        #  ||[navigator]       ||| Statistics                         |  |
        #  ||                  |||                                    |  |
        #  ||                  ||| Selector                           |  |
        #  |+------------------+|+------------------------------------+  |
        #  +--------------------+----------------------------------------+
        plotBar = QHBoxLayout()
        plotBar.addWidget(self._dataInspector)
        plotBar.addWidget(self._trainingBox)
        plotBar.addWidget(self._pltOriginal)
        plotBar.addWidget(self._pltAdversarial)

        buttonBar = QHBoxLayout()
        buttonBar.addWidget(self._buttonCreateModel)
        buttonBar.addWidget(self._buttonTrainModel)
        buttonBar.addWidget(self._buttonLoadModel)
        buttonBar.addWidget(self._buttonSaveModel)
        buttonBar.addWidget(self._buttonResetModel)
        buttonBar.addWidget(self._buttonPlotModel)
        buttonBar.addWidget(self._buttonShowExample)

        layout = QVBoxLayout()
        layout.addLayout(plotBar)
        layout.addLayout(buttonBar)
        self.setLayout(layout)

    def setImage(self, image: Imagelike) -> None:
        """Set the image for this :py:class:`FacePanel`. This
        will initiate the processing of this image using the
        current tools.
        """
        self.setData(Image.as_data(image))

    def setData(self, data: Data) -> None:
        """Set the data to be processed by this :py:class:`FacePanel`.
        """
        # set data for the dataView - this is redundant if data is set
        # from the toolbox (as the dataView also observes the toolbox),
        # but it is necessary, if setData is called independently.
        self._dataView.setData(data)

        # FIXME[todo]: generate adversarial example.

    def setToolbox(self, toolbox: Toolbox) -> None:
        """Set a new Toolbox.
        We are only interested in changes of the input data.
        """
        self._dataInspector.setToolbox(toolbox)
        # self._dataView.setToolbox(toolbox)
        self.setData(toolbox.input_data if toolbox is not None else None)

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        # pylint: disable=invalid-name
        """The FacePanel is a Toolbox.Observer. It is interested
        in input changes and will react with applying face recognition
        to a new input image.
        """
        if change.input_changed:
            self.setData(toolbox.input_data)

    # FIXME[old]
    # FIXME[hack]: no quotes!
    def setController(self, controller: 'AdversarialExampleController') -> None:
        self._controller = controller
        self._buttonCreateModel.clicked.connect(controller.create_model)
        self._buttonTrainModel.clicked.connect(controller.train_model)
        self._buttonLoadModel.clicked.connect(controller.load_model)
        self._buttonSaveModel.clicked.connect(controller.save_model)
        self._buttonResetModel.clicked.connect(controller.reset_model)
        self.observe(controller)

    def _enableComponents(self, running=False):
        print(f"enable components: {running}")
        available = self._controller is not None and not running
        self._buttonCreateModel.setEnabled(not running)
        for w in (self._buttonTrainModel,
                  self._buttonLoadModel, self._buttonSaveModel,
                  self._buttonPlotModel, 
                  self._buttonShowExample):
            w.setEnabled(available)

    def _onShowExample(self):
        if self._controller is None:
            self._pltOriginal.noData()
            self._pltAdversarial.noData()
        else:
            example_data, example_label, example_prediction = \
                self._controller.get_example()
            with self._pltOriginal as ax:
                ax.imshow(example_data[:,:,0], cmap='Greys_r')
                ax.set_title(f"Label = {example_label.argmax()}, "
                             f"Prediction = {example_prediction.argmax()}")

            adversarial_data, adversarial_prediction = \
                self._controller.get_adversarial_example()
            with self._pltAdversarial as ax:
                ax.imshow(adversarial_data[:,:,0], cmap='Greys_r')
                ax.set_title(f"Prediction = {adversarial_prediction.argmax()}")

    def adversarialControllerChanged(self, controller, change):
        if 'busy_changed' in change:
            self._enableComponents(controller.busy)
