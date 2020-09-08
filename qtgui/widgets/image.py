# standard imports
import sys

# Generic imports
import numpy as np
import logging

# Qt imports
from PyQt5.QtCore import Qt, QPoint, QSize, QRect
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import (QImage, QPainter, QPen, QTransform,
                         QKeyEvent, QMouseEvent)
from PyQt5.QtWidgets import QWidget, QMenu, QAction, QSizePolicy

# toolbox imports
from toolbox import Toolbox
from tools.activation import Engine as ActivationEngine
from datasource import Data, Metadata
from util.image import BoundingBox, PointsBasedLocation, Region

from dltb.base.image import ImageTool
from dltb.util.image import imresize, imwrite

# GUI imports
from ..utils import QObserver, protect

# logging
LOG = logging.getLogger(__name__)


# FIXME[todo]: add docstrings!
# FIXME[todo]: rename: QImageView is an old name ...

class QImageView(QWidget, QObserver, Toolbox.Observer,
        ActivationEngine.Observer, qobservables={
            Toolbox: {'input_changed'},
            ImageTool: {'image_changed'},
            ActivationEngine: {'activation_changed', 'input_changed',
                               'unit_changed'}}):
    """An experimental class to display images using the ``QImage``
    class.  This may be more efficient than using matplotlib for
    displaying images.

    Attributes
    ----------
    _data: Data
        The data object from which the image is taken (may be `None`)
    _image: QImage
        The image displayed
    _overlay: QImage
        Overlay for displaying on top of the image
    _show_raw: bool
        A flag indicating whether this QImageView will show
        the raw input data, or the data actually fed to the network.
    _showMetadata: bool
        A flag indicating if metadata should be shown if available.
    """
    """A :py:class:`QImageView` to display the input image of the
    Toolbox or ActivationEngine.


    **Toolbox.Observer:**
    The :py:class:`QImageView` can observe a
    :py:class:`Toolbox` to always display the
    current input image. If a 'input_changed' is reported, the
    QImageView is updated to display the new input image.


    **ActivationEngine.Observer:**
    FIXME[todo]: Currently not implemented!


    Attributes
    ----------
    _toolbox: Toolbox
        The toolbox we want to observe.
    _processed: bool
        A flag indicating if the raw or the preprocessed input
        data should be shown.

    _activation: ActivationEngine
        The activation Engine observed by this QImageView.

    _image: QImage = None
    _raw: np.ndarray = None
    _marks: list = None
    _overlay: QImage = None
    _show_raw: bool = False
    _keepAspect: bool = True

    _showMetadata: bool = True
    _metadata: Metadata = None
    _metadata2: Metadata = None
    _regions: list

    _contextMenu: QMenu
        A context menu allowing to save the current image.

    Signals
    -------
    """
    keyPressed = pyqtSignal(int)
    modeChanged = pyqtSignal(bool)

    def __init__(self, toolbox: Toolbox = None,
                 activation: ActivationEngine = None, **kwargs) -> None:
        """Construct a new :py:class:`QImageView`.

        Arguments
        ---------
        parent: QWidget
        """
        super().__init__(**kwargs)
        self._raw: np.ndarray = None
        self._show_raw = False

        self._data = None
        self._image = None
        self._marks = None
        self._receptiveField = None
        self._overlay = None
        self._metadata = None
        self._metadata2 = None
        self._regions = []

        self._overlay = None

        self._keepAspect = True
        self._showMetadata = True

        self._processed = False  

        self.modeChanged.connect(self.onModeChanged)

        self._toolbox: Toolbox = None
        self.setToolbox(toolbox)

        self._activation = None
        self.setActivationEngine(activation)
        
        #
        # Prepare the context Menu
        #
        self._contextMenu = QMenu(self)
        self._contextMenu.addAction(QAction('Info', self))

        aspectAction = QAction('Keep Aspect Ratio', self)
        aspectAction.setCheckable(True)
        aspectAction.setChecked(self._keepAspect)
        aspectAction.setStatusTip('Keep original aspect ratio of the image')
        aspectAction.toggled.connect(self.onAspectClicked)
        self._contextMenu.addAction(aspectAction)
        
        self._contextMenu.addSeparator()
        saveAction = QAction('Save image', self)
        saveAction.setStatusTip('save the current image')
        saveAction.triggered.connect(self.onSaveClicked)
        self._contextMenu.addAction(saveAction)
        self._contextMenu.addAction(QAction('Save image as ...', self))

        # set button context menu policy
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.onContextMenu)

        # set the size policy
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        #sizePolicy.setHeightForWidth(self.assetsListWidget.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by 'Tab' key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)

    #@pyqtSlot(QPoint)
    @protect
    def onContextMenu(self, point):
        # show context menu
        print(f"QImageView.onContextMenu({point}) [{type(point)}]")
        self._contextMenu.exec_(self.mapToGlobal(point))

    #@pyqtSlot(bool)
    @protect
    def onAspectClicked(self, checked):
        self.keepAspectRatio = checked

    @protect
    def onSaveClicked(self, checked):
        if (self._raw is not None and
            self._metadata is not None and
            self._metadata.has_attribute('basename')):
            # write the file
            imwrite(self._metadata.get_attribute('basename'), self._raw)

    @property
    def keepAspectRatio(self) -> bool:
        return self._keepAspect

    @keepAspectRatio.setter
    def keepAspectRatio(self, flag: bool) -> None:
        if self._keepAspect != flag:
            self._keepAspect = flag
            self.update()

    @property
    def showMetadata(self) -> bool:
        return self._showMetadata

    @showMetadata.setter
    def showMetadata(self, flag: bool) -> None:
        if self._showMetadata != flag:
            self._showMetadata = flag
            self.update()

    def setData(self, data: Data, attribute='data', index: int=0) -> None:
        """Set the data to be displayed by this :py:class:`QImageView`.
        """
        LOG.debug("QImageView.setData(%s, attribute='%s', index=%d): "
                  "new=%s, none=%s", data, attribute, index,
                  data is not self._data, data is None)
        data = data[index] if data is not None and data.is_batch else data
        self._data = data
        self.setImage(None if data is None else getattr(data, attribute))

    def getImage(self) -> np.ndarray:
        return self._raw

    def setImage(self, image: np.ndarray) -> None:
        """Set the image to display.
        """
        LOG.debug("QImageView.setImage(%s)", image is not None and image.shape)
        self._raw = image
        self._marks = []
        self._regions = []
        if image is not None:
            # To construct an 8-bit monochrome QImage, we need a
            # 2-dimensional, uint8 numpy array
            if image.ndim == 4:
                image = image[0]

            img_format = QImage.Format_Grayscale8
            bytes_per_line = image.shape[1]

            if image.ndim == 3:
                # three channels -> probably rgb
                if image.shape[2] == 3:
                    img_format = QImage.Format_RGB888
                    bytes_per_line *= 3
                else:
                    image = image[:, :, 0]

            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = image.copy()

            self._image = QImage(image,
                                 image.shape[1], image.shape[0],
                                 bytes_per_line, img_format)
            # self.resize(self._image.size())
        else:
            self._image = None

        # self.updateGeometry()
        self.update()

    def minimumSizeHint(self):
        # FIXME[hack]: this will change the size of the widget depending
        # on the size of the image, probably not what we want ...
        # return QSize(-1,-1) if self._image is None else self._image.size()
        return QSize(100, 100)

    def sizeHint(self):
        return QSize(1000, 1000)

    def setMask(self, mask):
        """Set a mask to be displayed on top of the actual image.

        Parameters
        ----------
        mask: numpy.ndarray
            If the mask has a different size than the actual image,
            it will be resized to fit the image.
        """
        if mask is None:
            self._overlay = None
        else:
            # We will use the mask values as alpha channel of a some
            # solid (red) QImage. This QImage can be displayed on top
            # of the actual image (which will be the background).
            #
            # Some care has to be taken here:
            # (1) The QImage should have the same size as the background
            #     image.
            # (2) To use the mask as the alpha channel of a QImage, it
            #     should be a contiguous uint8 numpy array of appropriate
            #     size.
            #
            # FIXME[problem/todo]: some of the resizing function seem
            # to change the datatype (from uint to float) and alse the
            # range (from [0,256] to [0.0,1.0]). This must not happen!
            # We need some stable and well-documented resizing API
            # from util.image (originally we used scipy.misc.imresize
            # here, which was well-behaved for our purposes, but which
            # has been deprecated and is no longer present in modern
            # versions of scipy)
            
            print(f"A: mask: {mask.flags['C_CONTIGUOUS']}, dtype={mask.dtype}, shape={mask.shape}, min={mask.min()}, max={mask.max()}")
            if not mask.flags['C_CONTIGUOUS'] or mask.dtype != np.uint8:
                mask = np.ascontiguousarray(mask, np.uint8)

            print(f"B: mask: {mask.flags['C_CONTIGUOUS']}, dtype={mask.dtype}, shape={mask.shape}, min={mask.min()}, max={mask.max()}")
            # print(f"B: mask: resize({mask.shape} -> {(self._image.height(), self._image.width())})")
            mask = imresize(mask, (self._image.height(), self._image.width()))
            print(f"C: mask: {mask.flags['C_CONTIGUOUS']}, dtype={mask.dtype}, shape={mask.shape}, min={mask.min()}, max={mask.max()}")
            mask = mask.astype(np.uint8)
            print(f"D: mask: {mask.flags['C_CONTIGUOUS']}, dtype={mask.dtype}, shape={mask.shape}, min={mask.min()}, max={mask.max()}")

            self._overlay = QImage(mask.shape[1], mask.shape[0],
                                   QImage.Format_ARGB32)
            self._overlay.fill(Qt.red)

            alpha = QImage(mask, mask.shape[1], mask.shape[0],
                           mask.shape[1], QImage.Format_Alpha8)
            painter = QPainter(self._overlay)
            painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            painter.drawImage(QPoint(), alpha)
            painter.end()
        self.update()

    def setReceptiveField(self, field: QRect, size: QSize = None) -> None:
        """Set the receptive field of a unit in an activation map.

        @param field
            A rectangle marking the boundaries of the rececptive field
        @param size
            The size of the reference system, relative to which the
            receptive field is described (e.g., the shape of the input
            layer of a :py:class:`Network`). If no reference size is
            is provided, the coordinates are assumed to refer to pixel
            positions in the image.
        """
        if self._image is None:
            self._receptiveField = None
        elif size is None or size == self._image.size():
            self._receptiveField = field
        else:
            # scale the receptive field to fit image coordinates
            ratioX = self._image.width() / size.width()
            ratioY = self._image.height() / size.height()

            point1 = QPoint(field.left() * ratioX, field.top() * ratioY)
            point2 = QPoint(field.right() * ratioX, field.bottom() * ratioY)
            self._receptiveField = QRect(point1, point2)

        self.update()

    def addMark(self, rect: QRect):
        """Mark a rectangle in the image.
        The rectangle provides the image coordinates to mark.
        """
        self._marks.append(rect)

    def setMetadata(self, metadata: Metadata, metadata2: Metadata = None):
        """Set metadata to be displayed in this View.
        """
        self._metadata = metadata
        self._metadata2 = metadata2
        self.update()

    def addRegion(self, region: Region):
        """Set metadata to be displayed in this View.
        """
        self._regions.append(region)
        self.update()

    @protect
    def paintEvent(self, event):
        """Process the paint event by repainting this Widget.

        Parameters
        ----------
        event : QPaintEvent
        """
        # FIXME[bug?]: this methods seems to be invoked quite often
        # - check if this is so and why!
        painter = QPainter()
        painter.begin(self)

        # Compute the transformation
        if self._image is not None:
            w = self._image.width()
            h = self._image.height()
            # scale maximally while maintaining aspect ratio
            w_ratio = self.width() / w
            h_ratio = self.height() / h
            if self._keepAspect:
                w_ratio = min(w_ratio, h_ratio)
                h_ratio = w_ratio
            # the rect is created such that it is centered on the
            # current widget pane both horizontally and vertically
            x = (self.width() - w * w_ratio) // 2
            y = (self.height() - h * h_ratio) // 2
            transform = QTransform()
            transform.translate(x, y)
            transform.scale(w_ratio, h_ratio)
            painter.setTransform(transform)

        self._drawImage(painter)
        self._drawReceptiveField(painter)
        self._drawMask(painter)
        self._drawMarks(painter)

        if self._image is not None and self.showMetadata:
            line_width = 4
            color = Qt.green
            pen = QPen(color)
            pen.setWidth(line_width * (1+max(self._image.width(),
                                             self._image.height())//400))
            painter.setPen(pen)

            for region in self._regions:
                self._drawRegion(painter, region)

            self._drawMetadata(painter, self._metadata, color=Qt.green)
            self._drawMetadata(painter, self._metadata2, color=Qt.red)
        painter.end()

    def _drawImage(self, painter: QPainter):
        """Draw current image into this ``QImageView``.

        Parameters
        ----------
        painter :   QPainter
        """
        if self._image is not None:
            painter.drawImage(QPoint(0, 0), self._image)

    def _drawMask(self, painter: QPainter):
        """Display the given image.

        Parameters
        ----------
        painter :   QPainter
        """
        if self._image is not None and self._overlay is not None:
            painter.drawImage(QPoint(0, 0), self._overlay)

    def _drawReceptiveField(self, painter: QPainter):
        if self._receptiveField is None:
            return

        pen_width = 4
        pen_color = Qt.green
        pen = QPen(pen_color)
        pen.setWidth(pen_width)
        painter.setPen(pen)
        painter.drawRect(self._receptiveField)

    def _drawMarks(self, painter: QPainter):
        """Draw marks on current image into this ``QImageView``.

        Parameters
        ----------
        painter :   QPainter
        """
        if not self._marks:
            return

        for mark in self._marks:
            if isinstance(mark, QRect):
                pen_width = 4
                pen_color = Qt.green
                pen = QPen(pen_color)
                pen.setWidth(pen_width)
                painter.setPen(pen)
                painter.drawRect(mark)

    def _drawMetadata(self, painter: QPainter, metadata: Metadata,
                      pen: QPen = None, line_width=4, color=Qt.green):
        if metadata is None or not metadata.has_regions():
            return

        if pen is None:
            pen = QPen(color)
            pen.setWidth(line_width * (1+max(self._image.width(),
                                             self._image.height())//400))
        painter.setPen(pen)

        for region in metadata.regions:
            self._drawRegion(painter, region)

    def _drawRegion(self, painter: QPainter, region: Region) -> None:
        location = region.location
        if isinstance(location, BoundingBox):
            painter.drawRect(location.x, location.y,
                             location.width, location.height)
        elif isinstance(location, PointsBasedLocation):
            for p in location.points:
                painter.drawPoint(p[0], p[1])

    #
    # Events
    #

    @protect
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """A mouse press toggles between raw and processed mode.
        """
        self.setMode(not self.mode())

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`QImageView` supports
        the following keys:

        r: toggle the keepAspectRatio flag
        """
        key = event.key()
        self.keyPressed.emit(key)

        if key == Qt.Key_R:
            self.keepAspectRatio = not self.keepAspectRatio
        elif key == Qt.Key_M:
            self.showMetadata = not self.showMetadata
        elif key == Qt.Key_Space:
            self.setMode(not self._processed)
        else:
            super().keyPressEvent(event)

    #
    # ImageTool.Observer
    #

    @protect
    def image_changed(self, tool, change) -> None:
        self.setImage(tool.image)

    #
    # Toolbox.Observer
    #

    @protect
    def toolbox_changed(self, toolbox: Toolbox, info: Toolbox.Change) -> None:
        if info.input_changed:
            # Just use the image from the Toolbox if no ActivationEngine
            # is available - otherwise we will use the image(s) from the
            # ActivationEngine (which will inform us via activation_changed ...)
            if self._activation is None:
                self._updateImage()
                self.setMask(None)
                
    #
    # ActivationEngine.Observer
    #

    def activation_changed(self, engine: ActivationEngine,
                           info: ActivationEngine.Change) -> None:
        """The :py:class:`QImageView` is interested in the
        input iamges, activations and units.
        """

        if info.input_changed:
            self._updateImage()

        return  #FIXME[old]
        if info.activation_changed: #  or info.unit_changed: FIXME[old]
            try:
                activation = engine.get_activation()
                unit = engine.unit
            except:
                activation = None
                unit = None

            # For convolutional layers add a activation mask on top of the
            # image, if a unit is selected
            if (activation is not None and unit is not None and
                activation.ndim > 1):
                # exclude dense layers
                from util.image import grayscaleNormalized
                activation_mask = grayscaleNormalized(activation[..., unit])
                self.setMask(activation_mask)
                field = engine.receptive_field
                if field is not None:
                    self.addMark(QRect(field[0][1], field[0][0],
                                       field[1][1]-field[0][1],
                                       field[1][0]-field[0][0]))
            else:
                self.setMask(None)

    #
    # Update
    #

    def _updateImage(self) -> None:
        """Set the image to be displayed either based on the current
        state of the Toolbox or the ActivationEngine. This will also
        take the current display mode (raw or processed) into account.
        """
        if self._activation is None:
            data = self._toolbox.input_data if self._toolbox else None
        elif self._processed:
            data = self._activation.input_data
        else:
            data = self._activation.raw_input_data

        self.setData(data)
        # FIXME[old]
        # self.setMetadata(self._toolbox and self._toolbox.input_metadata)

    @pyqtSlot(bool)
    def onModeChanged(self, processed: bool) -> None:
        """The display mode was changed. There are two possible modes:
        (1) processed=False (raw): the input is shown as provided by the source
        (2) processed=True: the input is shown as presented to the network

        Arguments
        ---------
        precessed: bool
            The new display mode (False=raw, True=processed).
        """
        self.setMode(processed)

    def mode(self) -> bool:
        return self._processed

    def setMode(self, processed: bool) -> None:
        if not self._toolbox:
            processed = False
        if processed != self._processed:
            self._processed = processed
            self.modeChanged.emit(processed)
            self._updateImage()
