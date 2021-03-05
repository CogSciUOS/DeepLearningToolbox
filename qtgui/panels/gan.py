"""
File: styltransfer.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
"""

# standard imports
import logging
import random

# third-party imports
import numpy as np

# Qt imports
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QPushButton, QSlider
from PyQt5.QtWidgets import QLabel, QGroupBox
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QSizePolicy

# toolbox imports
#import dltb
#from models.styletransfer import StyletransferTool

from dltb.base.prepare import Preparable
from dltb.base.busy import BusyObservable
from dltb.base.image import ImageObservable
from dltb.tool.generator import ImageGeneratorWorker
from dltb.util.importer import Importer
from dltb.thirdparty import implementations

# GUI imports
from .panel import Panel
from ..adapter import QAdaptedComboBox
from ..widgets.image import QImageView
from ..widgets.features import QFeatureView, QFeatureInfo
from ..utils import QBusyWidget, QPrepareButton, QObserver, protect

# logging
LOG = logging.getLogger(__name__)


class GANPanel(Panel, QObserver, qobservables={
        ImageObservable: {'image_changed', 'data_changed'},
        Preparable: {'state_changed'},
        BusyObservable: {'busy_changed'}}):
    """The :py:class:`GANPanel` provides a graphical interface provides
    controls that allow to run different forms of GANs. The panel
    contains the following groups of controls:

    * Selection and initialization:

    * Info and Display:

    * Transition: Interpolate between different features.


    Attributes
    ----------

    _ganName: str
        Name of the currently selected GAN module (one of the keys of
        of the `GANModules` dictionary).

    _ganClass: str
        The class from which the GAN can be instantiated.

    _gan: GAN
        The currently selected GAN. An instance of the class
        named by `_ganName`. `None` means that the class was not yet
        initialized.

    _worker:
        A :py:class:`Worker` object that initiates the actual
        generation and informs the GUI once images are available.
    """

    # initialize the GAN classes dictionary
    GANClasses = dict()
    for implementation in implementations('ImageGAN'):
        module, name = implementation.rsplit('.', maxsplit=1)
        if name in GANClasses or True:
            name += f" ({module.rsplit('.', maxsplit=1)[1]})"
        GANClasses[name] = implementation

    _ganName: str = None
    _ganClass: type = None
    _gan = None

    def __init__(self, **kwargs) -> None:
        """Initialize the :py:class:`StyletransferPanel`.
        """
        LOG.info("GANPanel: Initializing")
        super().__init__(**kwargs)
        self._gan = None
        self._busy = False
        self._features = None
        self._features1 = None
        self._features2 = None
        self._worker = ImageGeneratorWorker()
        self._importer = Importer()
        self._initUI()
        self._initLayout()
        self.observe(self._worker)
        self.observe(self._importer)
        self.setGan(None)

    def _initUI(self) -> None:
        """Initialize craphical components of the :py:class:`GANPanel`
        """

        #
        # GAN Initialization widgets
        #

        # A QComboBox for selecting the GAN class
        self._ganModuleLabel = QLabel("Module:")
        self._ganSelector = QAdaptedComboBox()
        self._ganSelector.setFromIterable(self.GANClasses.keys())
        self._ganSelector.currentTextChanged. \
            connect(self.onGanChanged)
        self._ganName = self._ganSelector.currentText()
        self._ganClass = self.GANClasses[self._ganName]

        # The "Initialize" button will trigger loading the GAN model
        self._initializeButton = QPushButton("Initialize")
        self._initializeButton.clicked.connect(self.onInitializeClicked)

        # A QComboBox for selecting a model (given the GAN class provides
        # more than one pre-trained model)
        self._ganModelLabel = QLabel("Model:")
        self._modelSelector = QAdaptedComboBox()
        self._modelSelector.currentTextChanged. \
            connect(self.onModelChanged)

        self._busyWidget = QBusyWidget()
        self._prepareButton = QPrepareButton()
        self._infoLabel = QLabel()

        self._debugButton = QPushButton("Debug")
        self._debugButton.clicked.connect(self.onDebugClicked)

        #
        # Generated image
        #

        # Image view for displaying generated images
        self._imageView = QImageView()

        # Feature view for displaying current features
        self._featureView = QFeatureView()
        self._featureInfo = QFeatureInfo()
        self._featureView.featuresChanged.\
            connect(self.updateImage)
        self._featureView.featuresChanged.\
            connect(self._featureInfo.updateFeatures)

        #
        # Interpolation
        #

        # Image view for displaying images "A" and "B" in a transition
        self._imageA = QImageView()
        self._imageB = QImageView()

        # Slider
        self._interpolationSlider = QSlider(Qt.Horizontal)
        self._interpolationSlider.setRange(0, 100)
        self._interpolationSlider.setSingleStep(1)
        self._interpolationSlider.valueChanged. \
            connect(self.onInterpolationChanged)

        # Button for randomly creating new images "A" and "B"
        self._randomButtonA = QPushButton("Random")
        self._randomButtonA.clicked.connect(self.onButtonClicked)
        self._randomButtonB = QPushButton("Random")
        self._randomButtonB.clicked.connect(self.onButtonClicked)

    def _initLayout(self) -> None:
        rows = QVBoxLayout()

        #
        # GAN Initialization panel
        #
        group = QGroupBox("GAN model")
        row = QHBoxLayout()
        row.addWidget(self._ganModuleLabel)
        self._ganModuleLabel.setSizePolicy(QSizePolicy.Fixed,
                                           QSizePolicy.Fixed)
        row.addWidget(self._ganSelector)
        row.addWidget(self._initializeButton)
        row.addWidget(self._ganModelLabel)
        self._ganModelLabel.setSizePolicy(QSizePolicy.Fixed,
                                          QSizePolicy.Fixed)
        row.addWidget(self._modelSelector)
        row.addWidget(self._prepareButton)
        row.addWidget(self._busyWidget)
        row.addWidget(self._infoLabel)
        row.addWidget(self._debugButton)
        group.setLayout(row)
        rows.addWidget(group)

        #
        # Generation
        #
        row = QHBoxLayout()
        row.addWidget(self._imageView)
        box = QVBoxLayout()
        box.addWidget(self._featureInfo)
        box.addStretch()
        row.addLayout(box)
        rows.addLayout(row)

        # Feature slider
        group = QGroupBox("Transition")
        row = QHBoxLayout()
        box = QVBoxLayout()
        box.addWidget(self._randomButtonA)
        box.addWidget(self._imageA)
        row.addLayout(box)
        #
        featureSlider = QVBoxLayout()
        featureSlider.addWidget(self._featureView)
        featureSlider.addWidget(self._interpolationSlider)
        row.addLayout(featureSlider, stretch=3)
        #
        box = QVBoxLayout()
        box.addWidget(self._randomButtonB)
        box.addWidget(self._imageB)
        row.addLayout(box)
        group.setLayout(row)
        rows.addWidget(group)

        self.setLayout(rows)

    @protect
    def onGanChanged(self, name: str) -> None:
        """A new GAN was selected in the `ganSelector` widget.
        If the corresponding GAN class is available (has already been
        imported), it will be instantiated and set as active GAN.
        Otherwise the import of the relevant module(s) will be
        initiated and the active GAN is set to `None`.

        Arguments
        ---------
        name:
            The name (identifier) of the new GAN.
        """
        LOG.info("GANPanel: GAN changed in selector: '%s' -> '%s'",
                 self._ganName, name)
        if name == self._ganName:
            return  # nothing changed

        # set the current GAN name
        self._gan = None
        self._ganName = name
        self._ganClass = self.GANClasses[self._ganName]
        self.update()

    @protect
    def onInitializeClicked(self, checked: bool) -> None:
        if self._gan is not None:
            return  # module already initialized

        # import class in background thread
        self._busyWidget.setBusyObservable(self._importer)
        self._importer.import_class(self._ganClass)
        self.update()

    def _initializeGan(self) -> None:
        """Import the StyleGAN module and update the interface.
        """
        if self._gan is not None:
            return  # GAN already initialized

        if self._ganClass is None:
            return  # nothing to initialize

        if isinstance(self._ganClass, str):
            if not self._importer.class_is_imported(self._ganClass):
                return  # class definition has not been imported
            self._ganClass = self._importer.imported_class(self._ganClass)

        self.setGan(self._ganClass())

    def setGan(self, gan) -> None:
        if self._gan is not None:
            self.unobserve(self._gan)

        self._gan = gan
        self._worker.generator = self._gan

        if self._gan is not None:
            self.observe(self._gan)

        self._prepareButton.setPreparable(self._gan)
        self._busyWidget.setBusyObservable(self._gan)

        # now update the interface to allow selecting one of the
        # pretrained stylegan models
        if hasattr(self._ganClass, 'models'):
            self._modelSelector.setFromIterable(self._ganClass.models)
            self._gan.model = self._modelSelector.currentText()
        else:
            self._modelSelector.clear()

        self.update()

    @protect
    def onModelChanged(self, name: str) -> None:
        if self._gan is None:
            return  # we can not change the model ...

        self._gan.model = name
        self.update()

    @protect
    def onButtonClicked(self, checked: bool) -> None:
        if self.sender() is self._randomButtonA:
            self._newRandomFeatures(index=0)
        elif self.sender() is self._randomButtonB:
            self._newRandomFeatures(index=1)

    def _newRandomFeatures(self, index: int = None) -> None:
        if index is None or index == 0 or self._features1 is None:
            seed1 = random.randint(0, 10000)
            self._features1 = self._gan.random_features(seed=seed1)

        if index is None or index == 1 or self._features2 is None:
            seed2 = random.randint(0, 10000)
            self._features2 = self._gan.random_features(seed=seed2)

        # image = self._gan.generate_image(self._features1)
        features = np.ndarray((3,) + self._features1.shape)
        features[0] = self._features1
        features[1] = self._features2
        scaled = self._interpolationSlider.value() / 100
        self._features = ((1-scaled) * self._features1 +
                          scaled * self._features2)
        features[2] = self._features
        self._worker.generate(features)
        self._featureView.setFeatures(self._features)
        self._featureInfo.setFeatures(self._features)

    @protect
    def onInterpolationChanged(self, value: int) -> None:
        """React to a change of the interpolation slider.
        """
        scaled = value / 100
        self._features = ((1-scaled) * self._features1 +
                          scaled * self._features2)
        self._worker.generate(self._features)  # will call gan.generate()
        self._featureView.setFeatures(self._features)
        self._featureInfo.setFeatures(self._features)

    def image_changed(self, observable: ImageObservable,
                      change: ImageObservable.Change) -> None:
        """Implementation of the :py:class:`ImageObserver` interface.
        This method is called when the image generator has finished
        creating a new image.
        """
        image = observable.image  # type Image
        if image.is_batch:
            self._imageA.setImagelike(image[0])
            self._imageB.setImagelike(image[1])
            self._imageView.setImagelike(image[2])
        else:
            self._imageView.setImagelike(image)  # FIXME[todo]: should be setImage, not setImageLike

    def preparable_changed(self, preparable: Preparable,
                           info: Preparable.Change) -> None:
        if preparable.prepared:
            self._infoLabel.setText(f"{self._gan.feature_dimensions} -> "
                                    f"?")
            if self._features1 is None:
                self._newRandomFeatures()
            self._gan.info()
        else:
            self._infoLabel.setText("")
            self._features1, self._features1 = None, None
            self._imageA.setImage(None)
            self._imageB.setImage(None)
            self._imageView.setImage(None)
            self._featureView.setFeatures(None)
            self._featureInfo.setFeatures(None)
        self.update()

    def busy_changed(self, importer: Importer,
                     change: Importer.Change) -> None:
        if importer is self._importer:
            if (self._gan is None and isinstance(self._ganClass, str) and
                    importer.class_is_imported(self._ganClass)):
                self._initializeGan()
            else:  # some error occured
                self.update()

    def update(self) -> None:
        """Update this :py:class:`GANPanel`.
        """

        initialized = self._gan is not None
        self._initializeButton.setVisible(not initialized)
        self._prepareButton.setVisible(initialized)
        self._ganSelector.setEnabled(not self._importer.busy)

        have_models = initialized and self._modelSelector.count() > 0
        self._ganModelLabel.setVisible(have_models)
        self._modelSelector.setVisible(have_models)

        if not initialized:
            self._initializeButton.setEnabled(not self._importer.busy)
        elif have_models:
            self._modelSelector.setEnabled(True)

        prepared = self._gan is not None and self._gan.prepared
        self._randomButtonA.setEnabled(prepared)
        self._randomButtonB.setEnabled(prepared)
        self._interpolationSlider.setEnabled(prepared)
        super().update()

    @pyqtSlot()
    def updateImage(self) -> None:
        self._worker.generate(self._features)

    @protect
    def onDebugClicked(self, checked: bool) -> None:
        print(f"DEBUG")
        print(f"gan: {self._gan}")
        if self._gan is not None:
            print(f" -prepared: {self._gan.prepared}")
            print(f" -busy: {self._gan.busy}")
            print(f" -models: {hasattr(self._gan, 'models')}")
        print(f"name: {self._ganName}")
        print(f"class: {self._ganClass}")
        self.update()
        if self._gan is not None:
            self._gan.info()
