"""
File: styltransfer.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
"""

# Generic imports
import sys
import logging
import random

# Qt imports
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QSlider
from PyQt5.QtWidgets import QLabel, QGroupBox
from PyQt5.QtWidgets import QStackedLayout, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QSizePolicy

# toolbox imports
#import dltb
#from models.styletransfer import StyletransferTool

from dltb.base.prepare import Preparable
from dltb.base.image import ImageObservable
from dltb.tool.generator import ImageGeneratorWorker

# GUI imports
from .panel import Panel
from ..adapter import QAdaptedComboBox
from ..widgets.image import QImageView
from ..utils import QBusyWidget, QPrepareButton, QObserver, protect

# logging
LOG = logging.getLogger(__name__)


class GANPanel(Panel, QObserver, qobservables={
        ImageObservable: {'image_changed', 'data_changed'},
        Preparable: {'state_changed'}}):
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

    _gan: GAN
        The currently selected GAN. An instance of the class
        named by `_ganName`. `None` means that the class was not yet
        initialized.

    _worker:
        A :py:class:`Worker` object that initiates the actual
        generation and informs the GUI once images are available.
    """

    GANModules = {
        'StyleGAN': 'dltb.thirdparty.stylegan.StyleGAN'
    }

    _ganName: str = None
    _gan = None

    def __init__(self, **kwargs) -> None:
        """Initialize the :py:class:`StyletransferPanel`.
        """
        super().__init__(**kwargs)
        self._gan = None
        self._worker = None
        self._busy = False
        self._initUI()
        self._initLayout()
        self.update()

    def _initUI(self) -> None:
        """Initialize craphical components of the :py:class:`GANPanel`
        """

        #
        # Initialization
        #

        # A QComboBox for selecting the GAN class (currently only StyleGAN)
        self._ganModuleLabel = QLabel("Module:")
        self._ganSelector = QAdaptedComboBox()
        self._ganSelector.setFromIterable(self.GANModules.keys())
        self._ganSelector.currentTextChanged. \
            connect(self.onGanChanged)

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

        #
        # Generated image
        #

        # Image view for displaying generated images
        self._imageView = QImageView()

        # Feature view for displaying current features
        self._featureView = QFeatureView()
        self._featureInfo = QFeatureInfo()

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
        # Initialization
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
        if name == self._ganName:
            return  # nothing changed
        self._ganName = name
        self._gan = None

    @protect
    def onInitializeClicked(self, checked: bool) -> None:
        if 'experiments.stylegan' in sys.modules:
            return  # module already initialized

        self._busyWidget.setBusy(True)
        self._busy = True
        # FIXME[hack]: we need a better initialization mechanism here!
        from threading import Thread
        Thread(target=self.doInitialize).start()
        self.update()

    def doInitialize(self):
        """Import the StyleGAN module and update the interface.
        """

        # import the StyleGAN class (this may take some time ...)
        # FIXME[hack]: create better import mechanism
        from dltb.thirdparty.stylegan import StyleGAN

        # now update the interface to allow selecting one of the
        # pretrained stylegan models
        self._modelSelector.setFromIterable(StyleGAN.models)
        self._busy = False
        self._busyWidget.setBusy(False)
        self.update()

    @protect
    def onModelChanged(self, name: str) -> None:
        # FIXME[hack]: create better import mechanism
        from dltb.thirdparty.stylegan import StyleGAN

        self._busyWidget.setBusy(True)
        if self._gan is not None:
            self.unobserve(self._gan)
        self._gan = StyleGAN(model=name)
        if self._gan is not None:
            self.observe(self._gan)

        if self._worker is None:
            self._worker = ImageGeneratorWorker(self._gan)
            self.observe(self._worker)
        else:
            self._worker.generator = self._gan

        self._prepareButton.setPreparable(self._gan)
        # self._gan.prepare()
        self._gan.info()
        self._busyWidget.setBusy(False)
        self.update()

    @protect
    def onButtonClicked(self, checked: bool) -> None:
        if self.sender() is self._randomButtonA:
            self._newRandomFeatures(index=0)
        elif self.sender() is self._randomButtonB:
            self._newRandomFeatures(index=1)

    def _newRandomFeatures(self, index: int = None) -> None:
        if index is None or index == 0:
            seed1 = random.randint(0, 10000)
            self._features1 = self._gan.random_features(seed=seed1)

        if index is None or index == 1:
            seed2 = random.randint(0, 10000)
            self._features2 = self._gan.random_features(seed=seed2)

        # image = self._gan.generate_image(self._features1)
        features = np.ndarray((3,) + self._features1.shape)
        features[0] = self._features1
        features[1] = self._features2
        scaled = self._interpolationSlider.value() / 100
        features[2] = (1-scaled) * self._features1 + scaled * self._features2
        self._worker.generate(features)
        self._featureView.setFeatures(features[2])
        self._featureInfo.setFeatures(features[2])

    @protect
    def onInterpolationChanged(self, value: int) -> None:
        """React to a change of the interpolation slider.
        """
        scaled = value / 100
        features = (1-scaled) * self._features1 + scaled * self._features2
        self._worker.generate(features)  # will call gan.generate()
        self._featureView.setFeatures(features)
        self._featureInfo.setFeatures(features)

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
            self._imageView.setImagelike(image)  # FIXME[todo]: should setImage

    def preparable_changed(self, preparable: Preparable,
                           info: Preparable.Change) -> None:
        if preparable.prepared:
            self._infoLabel.setText(f"{self._gan.feature_dimensions} -> "
                                    f"?")
            if self._features1 is None:
                self._newRandomFeatures()
        else:
            self._infoLabel.setText("")
            self._features1, self._features1 = None, None
        self.update()

    def update(self) -> None:
        """Update this :py:class:`GANPanel`.
        """

        # update GAN class initialization
        ganId = self._ganSelector.currentText()
        if ganId in self.GANModules:
            module, name = self.GANModules[ganId].rsplit('.', maxsplit=1)
            initialized = module in sys.modules
            self._initializeButton.setEnabled(not initialized and
                                              not self._busy)
            self._initializeButton.setVisible(not initialized)
        else:
            initialized = False
            self._initializeButton.setVisible(False)
            self._initializeButton.setEnabled(False)

        self._ganSelector.setEnabled(not self._busy)

        if not initialized:
            self._ganModelLabel.setVisible(False)
            self._modelSelector.setVisible(False)
            self._prepareButton.setVisible(False)
        elif self._modelSelector.count() > 0:
            self._ganModelLabel.setVisible(True)
            self._modelSelector.setVisible(True)
            self._modelSelector.setEnabled(initialized)
            self._prepareButton.setVisible(True)
        else:
            self._ganModelLabel.setVisible(False)
            self._modelSelector.setVisible(False)
            self._prepareButton.setVisible(False)

        haveGan = self._gan is not None
        enabled = haveGan and not self._busy and self._gan.prepared
        self._randomButtonA.setEnabled(enabled)
        self._randomButtonB.setEnabled(enabled)
        self._interpolationSlider.setEnabled(enabled)
        super().update()


import numpy as np

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy

class QFeatureView(QWidget):

    STYLE_HORIZONTAL = 1
    STYLE_VERTICAL = 2

    style = STYLE_HORIZONTAL

    def __init__(self, style: int = STYLE_HORIZONTAL, **kwargs) -> None:
        """Initialize the :py:class:`StyletransferPanel`.
        """
        super().__init__(**kwargs)
        self._features = None
        self._style = style
        if style == self.STYLE_HORIZONTAL:
            self.setSizePolicy(QSizePolicy.MinimumExpanding,
                               QSizePolicy.Preferred)
        else:
            self.setSizePolicy(QSizePolicy.Preferred,
                               QSizePolicy.MinimumExpanding)

    def setFeatures(self, features: np.ndarray) -> None:
        self._features = features
        self.update()

    def minimumSizeHint(self):
        """The minimum size hint.

        Returns
        -------
        QSize : The minimal size of this :py:class:`QFeatureView`
        """
        features = 200 if self._features is None else len(self._features)
        return QSize(features if self._style == self.STYLE_HORIZONTAL else 200,
                     200 if self._style == self.STYLE_HORIZONTAL else features)

    def paintEvent(self, event) -> None:
        """Process the paint event by repainting this Widget.

        Parameters
        ----------
        event : QPaintEvent
        """
        if self._features is not None:
            qp = QPainter()
            qp.begin(self)
            self._drawWidget(qp, event.rect())
            qp.end()

    def _drawWidget(self, qp, rect) -> None:
        """Draw a given portion of this widget.
        Parameters
        ----------
        qp : QPainter
        rect : QRect
        """
        pen = QPen(Qt.red)
        pen_width = 1
        pen.setWidth(pen_width)

        width = self.width()
        height = self.height()

        if self._style == self.STYLE_HORIZONTAL:
            center_y = height // 2
            stretch_y = center_y / np.abs(self._features).max()

            stretch_x = width / len(self._features)
            for x, feature in enumerate(self._features):
                pos_x = int(x*stretch_x)
                pos_y = int(center_y + feature * stretch_y)
                qp.drawLine(pos_x, center_y, pos_x, pos_y)
        else:
            width = self.width()
            center_x = width // 2
            stretch_x = center_x / np.abs(self._features).max()

            stretch_y = height / len(self._features)
            for y, feature in enumerate(self._features):
                pos_x = int(center_x + feature * stretch_x)
                pos_y = int(y*stretch_y)
                qp.drawLine(center_x, pos_y, pos_x, pos_y)


from PyQt5.QtWidgets import QLabel, QGridLayout


class QFeatureInfo(QWidget):

    def __init__(self, **kwargs) -> None:
        """Initialize the :py:class:`QFeatureInfo`.
        """
        super().__init__(**kwargs)
        self._initUI()
        self.setFeatures(None)

    def _initUI(self) -> None:
        grid = QGridLayout()
        # Dimensionality
        grid.addWidget(QLabel("Dimensionality"), 0, 0)
        self._dimensionality = QLabel()
        grid.addWidget(self._dimensionality, 0, 1)
        # Min/max
        grid.addWidget(QLabel("min/max"), 1, 0)
        self._minmax = QLabel()
        grid.addWidget(self._minmax, 1, 1)
        # Min/max
        grid.addWidget(QLabel("L2-norm"), 2, 0)
        self._l2norm = QLabel()
        grid.addWidget(self._l2norm, 2, 1)
        # density
        grid.addWidget(QLabel("density"), 3, 0)
        self._density = QLabel()
        grid.addWidget(self._density, 3, 1)

        # add the layout
        self.setLayout(grid)

    def setFeatures(self, features: np.ndarray) -> None:
        self._features = features
        if features is None:
            self._dimensionality.setText('')
            self._minmax.setText('')
            self._l2norm.setText('')
            self._density.setText('')
        else:
            self._dimensionality.setText(f"{len(features)}")
            self._minmax.setText(f"{features.min():.4f}/{features.max():.4f}")
            l2norm = np.linalg.norm(features)
            self._l2norm.setText(f"{l2norm:.2f}")
            # Gaussian density
            density = 1/np.sqrt(2*np.pi) * np.exp(-0.5 * l2norm**2)
            self._density.setText(f"{density:1.2e}")
        self.update()
