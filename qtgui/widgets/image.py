"""QWidgets for displaying images.
"""

# standard imports
from typing import Union, List, Iterable, Tuple
import logging

# Generic imports
import numpy as np

# Qt imports
from PyQt5.QtCore import Qt, QPoint, QPointF, QSize, QRect
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPainter, QPen, QTransform
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QPaintEvent
from PyQt5.QtWidgets import QWidget, QMenu, QAction, QSizePolicy, QVBoxLayout
from PyQt5.QtWidgets import QToolTip

# toolbox imports
from dltb.base.data import Data
from dltb.base.meta import Metadata
from dltb.base.image import Image, Imagelike, ImageObservable
from dltb.base.image import BoundingBox, PointsBasedLocation, Region, Landmarks
from dltb.tool.image import ImageTool
from dltb.util.image import imresize, imwrite, grayscaleNormalized
from toolbox import Toolbox

# GUI imports
from ..utils import QObserver, protect
from .navigation import QIndexControls

# logging
LOG = logging.getLogger(__name__)


# FIXME[todo]: add docstrings!
# FIXME[todo]: rename: QImageView is an old name ...

def imageToQImage(image: Imagelike) -> QImage:

    if image is None:
        return None

    image = Image.as_array(image)

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
        if image.max() < 2:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    image = image.copy()

    return QImage(image, image.shape[1], image.shape[0],
                  bytes_per_line, img_format)


class QImageView(QWidget, QObserver, Toolbox.Observer, qobservables={
        Toolbox: {'input_changed'},
        ImageObservable: {'image_changed'}}):
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
    _toolTipActive : bool
        A flag indicating whether tooltips shoould be shown.

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
    modeChanged = pyqtSignal(bool)


    Toolbox interface
    -----------------

    A :py:class:`QImageView` to display the input image of the
    Toolbox.

    **Toolbox.Observer:**
    The :py:class:`QImageView` can observe a
    :py:class:`Toolbox` to always display the
    current input image. If a 'input_changed' is reported, the
    QImageView is updated to display the new input image.


    ActivationTool interface
    ------------------------

    **ActivationEngine.Observer:**
    FIXME[todo]: Currently not implemented!


    ImageGenerator Observer
    -----------------------
    The QImageView may be assigned to an :py:class:`ImageGenerator`.
    If this is the case, the :py:class:`ImageGenerator` can provide
    information on how the image is used by that tool and that
    information can be visualized by the :py:class:`QImageView`.

    """
    modeChanged = pyqtSignal(bool)

    currentRegionChanged = pyqtSignal(int)

    def __init__(self, toolbox: Toolbox = None, **kwargs) -> None:
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
        self._activation = None
        self._marks = None
        self._receptiveField = None
        self._overlay = None
        self._metadata = None
        self._metadata2 = None
        self._regions = []
        self._currentRegion = -1  # -1 means no region selected

        self._overlay = None

        self._keepAspect = True
        self._showMetadata = True

        self._offset = QPointF(0, 0)
        self._zoom = 1.0

        self._mouseStartPosition = None

        self._processed = False
        self._toolTipActive = False
        self.setToolTip(False)

        self._toolbox: Toolbox = None
        self.setToolbox(toolbox)

        self._imageTool = None
        self._imageToolRect = None
        self._imageToolSize = None
        self._imageToolTransform = None

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
        #
        # QSizePolicy.MinimumExpanding: The sizeHint() is minimal, and
        # sufficient. The widget can make use of extra space, so it
        # should get as much space as possible.
        sizePolicy = QSizePolicy(QSizePolicy.MinimumExpanding,
                                 QSizePolicy.MinimumExpanding)
        # sizePolicy.setHeightForWidth(True)
        self.setSizePolicy(sizePolicy)

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by 'Tab' key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)

    @staticmethod
    def minimumSizeHint():
        return QSize(100, 100)  # default is QSize(-1, -1)

    @staticmethod
    def sizeHint():
        return QSize(200, 200)

    @staticmethod
    def heightForWidth(width: int) -> int:
        return width

    @staticmethod
    def hasHeightForWidth() -> bool:
        return True

    @protect
    def onContextMenu(self, point: QPoint):
        """React to the display of a context menu.

        Arguments
        ---------
        point:
            The location where the context menu is to be displayed, in
            widgets coordinates (relative to the left upper corner
            of this :py:class:`QImageView`).
        """
        # show context menu
        self._contextMenu.exec_(self.mapToGlobal(point))

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

    def setToolTip(self, active: bool = True) -> None:
        """Turn on/off the tooltips for this Widget.
        The tooltip will display the index and value for the matrix
        at the current mouse position.

        Parameters
        ----------
        active: bool
            Will turn the tooltips on or off.
        """
        self._toolTipActive = active
        self.setMouseTracking(self._toolTipActive)

        if not self._toolTipActive:
            QToolTip.hideText()

    def imageTool(self) -> ImageTool:
        """The :py:class:`ImageTool` used by this :py:class:`QImageView`.
        """
        return self._imageTool

    def setImageTool(self, imageTool: ImageTool) -> None:
        """Set an :py:class:`ImageTool` for this :py:class:`QImageView`.

        Arguments
        ---------
        imageTool:
            The :py:class:`ImageTool` to be used by this
            :py:class:`QImageView`.  `None` means that no
            :py:class:`ImageTool` is used.
        """
        self._imageTool = imageTool
        self._updateImageTool()

    def _updateImageTool(self) -> None:
        """Update the :py:class:`ImageTool`
        """
        imageTool, image = self._imageTool, self._raw

        if imageTool is None or image is None:
            self._imageToolRect = None
            self._imageToolSize = None
            self._imageToolTransform = None
        else:
            rectangle = imageTool.region_of_image(image)
            self._imageToolRect = \
                QRect(rectangle[0], rectangle[1],
                      rectangle[2]-rectangle[0], rectangle[3]-rectangle[1])
            size = QSize(*imageTool.size_of_image(image))
            imageSize = QSize(image.shape[1], image.shape[0])
            self._imageToolSize = size

            # imageToolTransform: map tool coordinates to image coordinates
            self._imageToolTransform = QTransform()
            self._imageToolTransform.translate(self._imageToolRect.x(),
                                               self._imageToolRect.y())
            self._imageToolTransform.scale(self._imageToolRect.width() /
                                           size.width(),
                                           self._imageToolRect.height() /
                                           size.height())
        self.update()

    def setData(self, data: Data, attribute='array', index: int = 0) -> None:
        """Set the data to be displayed by this :py:class:`QImageView`.
        """
        LOG.debug("QImageView.setData(%s, attribute='%s', index=%d): "
                  "new=%s, none=%s", data, attribute, index,
                  data is not self._data, data is None)
        data = data[index] if data is not None and data.is_batch else data
        if data is self._data:
            return  # nothing changed
        self._data = data
        self.setImage(None if data is None else getattr(data, attribute))

    def setImagelike(self, image: Imagelike) -> None:
        array = Image.as_array(image, dtype=np.uint8)
        self.setImage(array)

    def getImage(self) -> np.ndarray:
        return self._raw

    def setImage(self, image: np.ndarray) -> None:
        """Set the image to display.
        """
        LOG.debug("QImageView.setImage(%s)", image is not None and image.shape)
        self._raw = image
        self._marks = []
        self._regions = []
        self._currentRegion = -1
        self._image = imageToQImage(image)
        if self._image is not None:
            pass
            # self.resize(self._image.size())

        # self.updateGeometry()
        self._updateImageTool()

    def setMask(self, mask):
        """Set a mask to be displayed on top of the actual image.

        Parameters
        ----------
        mask: numpy.ndarray
            If the mask has a different size than the actual image,
            it will be resized to fit the image.
            The mask is assumed to be in the (height,width) format.
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

            print(f"A: mask: {mask.flags['C_CONTIGUOUS']}, dtype={mask.dtype}, shape={mask.shape}, min={mask.min()}, max={mask.max()}")
            if not mask.flags['C_CONTIGUOUS'] or mask.dtype != np.uint8:
                mask = np.ascontiguousarray(mask, np.uint8)

            print(f"B: mask: {mask.flags['C_CONTIGUOUS']}, dtype={mask.dtype}, shape={mask.shape}, min={mask.min()}, max={mask.max()}")
            # print(f"B: mask: resize({mask.shape} -> {(self._image.height(), self._image.width())})")
            mask = imresize(mask, (self._image.width()-5, self._image.height()-5))
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

    def addRegion(self, region: Region) -> None:
        """Set metadata to be displayed in this View.
        """
        self._regions.append(region)
        self.update()

    def addLandmarks(self, landmarks: Landmarks) -> None:
        """Add landmarks to be displayed on the current image.
        """
        self._regions.append(Region(landmarks))
        self.update()

    def invalidateRegion(self, index: int = None) -> None:
        """Invalidate an image in this :py:class:`QMultiImageView`.
        Invalid images will be displayed in a different way as
        valid images.

        Arguments
        ---------
        index:
            An index identifying the image to be invalidated. If no index is
            provided, the currently selected image is used. If there is
            no such index, the method will do nothing.
        """
        if index is None:
            index = self._currentRegion
        if not self._regions or not 0 <= index < len(self._regions):
            return

        region = self._regions[index]
        region.invalid = not getattr(region, 'invalid', False)
        self.update()

    def transform(self) -> QTransform:
        if self._image is None:
            return None

        w = self._image.width()
        h = self._image.height()
        # scale maximally while maintaining aspect ratio
        w_ratio = self.width() / w
        h_ratio = self.height() / h
        if self._keepAspect:
            w_ratio = min(w_ratio, h_ratio)
            h_ratio = w_ratio
        w_ratio *= self._zoom
        h_ratio *= self._zoom
        # the rect is created such that it is centered on the
        # current widget pane both horizontally and vertically
        x = (self.width() - w * w_ratio) // 2
        y = (self.height() - h * h_ratio) // 2
        transform = QTransform()
        transform.translate(x + self._offset.x(), y + self._offset.y())
        transform.scale(w_ratio, h_ratio)
        return transform

    def setZoom(self, zoom: float) -> None:
        self._zoom = zoom
        self.update()

    @protect
    def paintEvent(self, event: QPaintEvent) -> None:
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
            painter.setTransform(self.transform())

        self._drawImage(painter)
        self._drawImageTool(painter)
        self._drawReceptiveField(painter)
        self._drawMask(painter)
        self._drawMarks(painter)
        self._drawMetadata(painter)

        painter.end()

    def _drawImage(self, painter: QPainter):
        """Draw current image into this ``QImageView``.

        Parameters
        ----------
        painter :   QPainter
        """
        if self._image is not None:
            painter.drawImage(QPoint(0, 0), self._image)

    def _drawImageTool(self, painter: QPainter):
        """Draw current image into this ``QImageView``.

        Parameters
        ----------
        painter :   QPainter
        """
        if self._imageToolRect is not None:
            pen_width = 4
            pen_color = Qt.blue
            pen = QPen(pen_color)
            pen.setWidth(pen_width)
            painter.setPen(pen)
            painter.drawRect(self._imageToolRect)

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

    def _drawMetadata(self, painter: QPainter) -> None:
        if self._image is not None and self.showMetadata:
            line_width = 4
            line_width *= (1+max(self._image.width(),
                                 self._image.height())//400)
            greenPen = QPen(Qt.green)
            greenPen.setWidth(line_width)
            redPen = QPen(Qt.red)
            redPen.setWidth(line_width)
            bluePen = QPen(Qt.blue)
            bluePen.setWidth(line_width)

            for index, region in enumerate(self._regions):
                painter.setPen(bluePen if index == self._currentRegion else
                               (redPen if getattr(region, 'invalid') else
                                greenPen))
                self._drawRegion(painter, region)

            self._drawMetadataHelper(painter, self._metadata, color=Qt.green)
            self._drawMetadataHelper(painter, self._metadata2, color=Qt.red)

    def _drawMetadataHelper(self, painter: QPainter, metadata: Metadata,
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
        """Draw a :py:class:`Region` on top of image. Currently supported
        regions are :py:class:`BoundingBox` (displayed as rectangle) and
        :py:class:`PointsBasedLocation` (displayed as points).
        """
        location = region.location
        if isinstance(location, BoundingBox):
            painter.drawRect(location.x, location.y,
                             location.width, location.height)
            if getattr(region, 'invalid', False):
                redPen = QPen(Qt.red)
                redPen.setWidth(5)
                painter.setPen(redPen)
                painter.drawLine(location.x1, location.y1,
                                 location.x2, location.y2)
                painter.drawLine(location.x2, location.y1,
                                 location.x1, location.y2)
        elif isinstance(location, PointsBasedLocation):
            for p in location.points:
                painter.drawPoint(p[0], p[1])

    #
    # Events
    #

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`QImageView` supports
        the following keys:

        space: toggle tool tips
        r: toggle the keepAspectRatio flag
        """
        key = event.key()

        # Space will toggle display of tooltips
        if key == Qt.Key_Space:
            self.setToolTip(not self._toolTipActive)
        elif key == Qt.Key_R:
            self.keepAspectRatio = not self.keepAspectRatio
        elif key == Qt.Key_M:
            self.showMetadata = not self.showMetadata
        elif key == Qt.Key_Space:
            self.setMode(not self._processed)
        elif key == Qt.Key_Delete:
            self.invalidateRegion()
        elif key == Qt.Key_I:
            LOG.info("image: %s, regions(%s): %s, "
                     "metadata: %s, metadata2: %s, "
                     "mode=%s, keep aspect ratio=%s, show meta data=%s, "
                     "tooltip=%s",
                     self._image is not None,
                     type(self._regions), self._regions and len(self._regions),
                     self._metadata is not None, self._metadata2 is not None,
                     self.mode(), self.keepAspectRatio, self.showMetadata,
                     self._toolTipActive)
        else:
            super().keyPressEvent(event)

    @protect
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """A mouse press toggles between raw and processed mode.
        """
        self.setMode(not self.mode())

        # remember start position (for dragging the image around)
        self._mouseStartPosition = event.pos()

        # get the mouse position im image coordinates
        transform = self.transform()
        if transform is None or not transform.isInvertible():
            return  # we can not determine the image coordinates
        inverted, _invertible = transform.inverted()
        imagePosition = inverted.map(event.pos())

        # check if mouse clicked into some region of interest
        for index, region in enumerate(self._regions):
            point = (imagePosition.x(), imagePosition.y())
            if point in region:
                self.setCurrentRegion(None if index == self._currentRegion
                                      else index)
                break
        else:
            self.setCurrentRegion(None)

    @protect
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """A mouse press toggles between raw and processed mode.
        """
        if self._mouseStartPosition is None:
            return  # something strange happened - we will ignore this
        self._offset += event.pos() - self._mouseStartPosition
        self._mouseStartPosition = None
        self.update()

    @protect
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._currentRegion >= 0:
            self.invalidateRegion()
        else:
            # Reset the image
            self._zoom = 1.0
            self._offset = QPointF(0, 0)
            self._mouseStartPosition = None
            self.update()

    def wheelEvent(self, event):
        """Process the wheel event. The mouse wheel can be used for
        zooming.

        Parameters
        ----------
        event : QWheelEvent
            The event providing the angle delta.
        """
        delta = event.angleDelta().y() / 120  # will be +/- 1
        # position = (self._offset + event.pos()) / self._zoom

        zoom = (1 + delta * 0.01) * self._zoom
        self.setZoom(zoom)

        # offset = (position * self._zoom) - event.pos()
        # self.setOffset(offset)

        # We will accept the event, to prevent interference
        # with the QScrollArea.
        event.accept()

    # Attention: The mouseMoveEvent() is only called for regular mouse
    # movements, if mouse tracking is explicitly enabled for this
    # widget by calling self.setMouseTracking(True).  Otherwise it may
    # be called on dragging.
    @protect
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Process mouse movements.  If tooltips are active, information on
        mouse pointer position are displayed.

        Parameters
        ----------
        event: QMouseEvent
            The mouse event, providing local and global coordinates.
        """
        if not self._toolTipActive:
            print("QImageView: mouseMoveEvent:", event.pos(),
                  bool(event.buttons() & Qt.LeftButton),
                  bool(event.buttons() & Qt.RightButton))
            if self._mouseStartPosition is not None:
                self._offset += event.pos() - self._mouseStartPosition
                self._mouseStartPosition = event.pos()
                self.update()
            return

        position = event.pos()
        size = self.size()
        text = f"Screen position: {(position.x(), position.y())}"
        text += f" in {size.width()}x{size.height()}"

        if self._image is None:
            text += "\nNo image"
        else:
            image_size = self._image.size()

        transform = self.transform()
        imagePosition = None
        if transform is None:
            text += "\nNo transformation"
        elif not transform.isInvertible():
            text += "\nTransformation not invertible"
        else:
            inverted, _invertible = transform.inverted()
            image_rect = QRect(0, 0, image_size.width(), image_size.height())
            projected_image_rect = transform.mapRect(image_rect)
            projected_image_size = projected_image_rect.size()
            text += (f"\nScreen image size: {projected_image_size.width()}x"
                     f"{projected_image_size.height()}")

            imagePosition = inverted.map(position)
            text += (f"\nScreen image position: "
                     f"({imagePosition.x()}, {imagePosition.y()})"
                     f" in {image_size.width()}x{image_size.height()}")

        if self._imageTool is None:
            text += "\nNo ImageTool"
        else:
            # toolTransform = self._imageToolTransformation
            text += "\nImageTool: "
            rect = self._imageToolRect
            if rect is None:
                text += "None"
            else:
                text += f"{rect}"
            size = self._imageToolSize
            if size is None:
                text += "None"
            else:
                text += f"{size}"

            tTransform = self._imageToolTransform
            if tTransform is not None and imagePosition is not None:
                iTransform, ok = self._imageToolTransform.inverted()
                if ok: 
                    toolPosition = iTransform.map(imagePosition)
                    text += f" at {toolPosition}"

        if self._overlay is None:
            text += "\nNo activation"
        else:
            text += f"\nActivation shape: {self._overlay.size()}"

        QToolTip.showText(event.globalPos(), text)

    #
    # ImageTool.Observer
    #

    @protect
    def image_changed(self, observable: ImageObservable,
                      change: ImageObservable.Change) -> None:
        """A new image is available.  Display that image in this
        :py:class:`QImagView`.
        """
        self.setImage(observable.image)

    #
    # Toolbox.Observer
    #

    @protect
    def toolbox_changed(self, toolbox: Toolbox, info: Toolbox.Change) -> None:
        if info.input_changed:
            self.setData(toolbox.input_data)

    #
    # ActivationEngine.Observer
    #

    # FIXME[old]
    def activation_changed(self, engine,  #: ActivationEngine,
                           info  # : ActivationEngine.Change
                           ) -> None:
        """The :py:class:`QImageView` is interested in the
        input iamges, activations and units.
        """

        if info.input_changed:
            self._updateImage()

        return  # FIXME[old]
        if info.activation_changed:  # or info.unit_changed: FIXME[old]
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

    def mode(self) -> bool:
        return self._processed

    def setMode(self, processed: bool) -> None:
        """The display mode was changed. There are two possible modes:
        (1) processed=False (raw): the input is shown as provided by the source
        (2) processed=True: the input is shown as presented to the network

        Arguments
        ---------
        precessed: bool
            The new display mode (False=raw, True=processed).
        """
        if not self._toolbox:
            processed = False
        if processed != self._processed:
            self._processed = processed
            self.modeChanged.emit(processed)
            self._updateImage()

    @pyqtSlot(int)
    def setCurrentRegion(self, index: int) -> None:
        """Set the index of the currently selected region.

        Arguments
        ---------
        index:
            The new index. A value of `-1` means that no region is
            selected.
        """
        if index != self._currentRegion:
            self._currentRegion = index
            self.currentRegionChanged.emit(index)
            self.update()

    @pyqtSlot(int)
    def updateRegion(self, index: int) -> None:
        """Set the index of the currently selected region.

        Arguments
        ---------
        index:
            The new index. A value of `-1` means that no region is
            selected.
        """
        self.update()


class QImageBatchView(QWidget):
    """A :py:class:`QWidget` to display a batch of images.
    """

    def __init__(self, images: Union[Data, np.ndarray] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self._images = None
        self.setImages(images)

    def _initUI(self) -> None:
        """Initialize the user interface. The essential components
        are a :py:class:`QImageView` to display an image and
        some navigation tool to select an image from the batch.
        """
        self._imageView = QImageView()

        self._indexControls = QIndexControls()
        self._indexControls.indexChanged.connect(self.onIndexChanged)

    def _layoutUI(self) -> None:
        """Layout the widget.
        """
        layout = QVBoxLayout()
        layout.addWidget(self._imageView)
        layout.addWidget(self._indexControls)
        self.setLayout(layout)

    def index(self) -> int:
        """The index of the currently selected image.
        """
        return self._indexControls.index()

    def setIndex(self, index: int) -> None:
        """Set the index of the image to display.
        """
        # Update the indexControls if necessary. This will
        # in turn emit the 'indexChanged' signal, which triggers
        # this function again.
        if index != self.index():
            self._indexControls.setIndex(index)
        elif (self._images is not None and
              index >= 0 and index < len(self._images)):
            self._imageView.setImage(self._images[index])
        else:
            self._imageView.setImage(None)

    def setImages(self, images: Union[Data, np.ndarray],
                  index: int = 0) -> None:
        """Set the images to be displayed in this
        :py:class:`QImageBatchView`.

        Arguments
        ---------
        images:
            The images belonging to the batch.

        index:
            The index of the image to display.
        """
        self._images = images
        if images is not None:
            self._indexControls.setEnabled(True)
            self._indexControls.setElements(len(images))
        else:
            self._indexControls.setEnabled(False)
            index = -1
        self.setIndex(index)
        self._indexControls.update()

    @protect
    def onIndexChanged(self, index: int) -> None:
        """A slot to react to changes of the batch index.
        """
        self.setIndex(index)


class QMultiImageView(QWidget):
    """Display multiple images. Generally, images are displayed in a grid,
    in simple cases images are displayed either horizontally (in a
    row) or vertically (in a column). The grid is devined by a
    pair of integers (rows, columns), where one of these values can
    be `None`, meaning the grid can dynamically grow into that direction.

    The size at which the individual images are displayed can be
    controlled by the property :py:prop:`imageSize`.

    The :py:class:`QMultiImageView` allows to select one image as
    current image.  When the current image is changed, the
    `currentImageChanged` signal is emitted with the index of the
    current image.  A value of `-1` means that no image is selected.


    Properties
    ----------
    grid: Tuple[int, int]
        The number of rows and columns displayed in this
        :py:class:`QMultiImageView`.
    orientation: Qt.Orientation
        The orientation of this :py:class:`QMultiImageView`. If
        `Qt.Horizontal`, the grid will grow horizontally, meaning
         images will be displayed in one (or multiple) rows, if
        `Qt.Vertical`, images will be displayed in columns.

    Readonly properties
    -------------------
    _qimages: List[QImage]
        A list of `QImage`s holding the images to be displayed.
    _qregions:
        A list of :py:class:`Region`s describing regions in a larger
        image from which the images displayed in this
        :py:class:`QMultiImageView` are extracted.  The regions
        may contain annotations which may influence how the corresponding
        image is displayed.
    """

    currentImageChanged = pyqtSignal(int)
    annotationsChanged = pyqtSignal(int)

    _qimages: List[QImage] = None

    def __init__(self, orientation: Qt.Orientation = Qt.Horizontal,
                 grid: Tuple[int, int] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if grid is not None:
            self._grid = grid
        else:
            self._grid = \
                (1, None) if orientation == Qt.Horizontal else (None, 1)
        self._imageSize = QSize(100, 100)
        self._spacing = 10
        self._qimages = []
        self._currentIndex = -1  # -1 means no image selected
        self._initLayout()

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by 'Tab' key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)

    def _initLayout(self) -> None:
        # set the size policy
        #
        # QSizePolicy.MinimumExpanding: The sizeHint() is minimal, and
        # sufficient. The widget can make use of extra space, so it
        # should get as much space as possible.
        self.setMinimumSize(self._imageSize.width() + self._spacing,
                            self._imageSize.height() + self._spacing)
        if self.orientation() == Qt.Horizontal:
            self.setSizePolicy(QSizePolicy.MinimumExpanding,
                               QSizePolicy.Fixed)
        else:
            self.setSizePolicy(QSizePolicy.Fixed,
                               QSizePolicy.MinimumExpanding)

    def count(self) -> int:
        """The number of images currently displayed in this
        :py:class:`QMultiImageView`.
        """
        return len(self._qimages)

    def rows(self) -> int:
        """The number of rows used for display in this
        :py:class:`QMultiImageView`.
        """
        if not self.count():
            return 0
        if self._grid[0] is not None:
            return self._grid[0]
        return -(-self.count() // self._grid[1])

    def columns(self) -> int:
        """The number of columns used for display in this
        :py:class:`QMultiImageView`.
        """
        if not self.count():
            return 0
        if self._grid[1] is not None:
            return self._grid[1]
        return -(-self.count() // self._grid[0])

    def grid(self) -> Tuple[int, int]:
        """The grid in terms of rows and columns used to display
        images in this :py:class:`QMultiImageView`.
        """
        return self._grid

    def setGrid(self, grid: Tuple[int, int]) -> None:
        """Set the grid in terms of rows and columns used to
         display images in this :py:class:`QMultiImageView`.
        Either rows or columns may be `None`, indicating that the
        display may grow as needed along this axis.
        """
        if grid != self._grid:
            self._grid = grid
            self._gridChanged()

    def orientation(self) -> Qt.Orientation:
        """The orientation of this :py:class:`QMultiImageView`. If
        `Qt.Horizontal`, images will be displayed in row, if
        `Qt.Vertical`, images will be displayed in column.
        """
        return Qt.Horizontal if self._grid[1] is None else Qt.Vertical

    def _gridPosition(self, index: int = None) -> Tuple[int, int]:
        """Get the grid position as (row, column) for the image with
        the given index.
        """
        if index is None:
            index = self._currentIndex
        if index is None:
            return -1, -1
        
        if self._grid[0] is not None:
            row, column = index % self._grid[0], index // self._grid[0]
        else:
            row, column = index // self._grid[1], index % self._grid[1]
        return row, column

    def _setGridPosition(self, row: int, column: int) -> None:
        """Set the index of the currently selected image.
        """
        if row == -1 or column == -1:
            index = -1
        elif self._grid[0] is not None:
            index = column * self._grid[0] + row
        else:
            index = row * self._grid[1] + column
        if index > self.count():
            index = self.count() - 1
        self.setCurrentImage(-1 if index == self._currentIndex else index)

    def _gridChanged(self) -> None:
        """The grid has changed. This implies that the size of this
        :py:class:`QMultiImageView` may have changed.
        """
        if self._grid[0] is not None:  # Qt.Horizontal
            width = ((-(-self.count() // self._grid[0])) *
                     (self._imageSize.width() + self._spacing))
            height = self._grid[0] * (self._imageSize.height() + self._spacing)
        else:  # Qt.Vertical
            width = self._grid[1] * (self._imageSize.width() + self._spacing)
            height = ((-(-self.count() // self._grid[1])) *
                      (self._imageSize.height() + self._spacing))
        self.setMinimumSize(width, height)
        self.resize(width, height)
        self.update()

    def imageSize(self) -> QSize:
        """The size at which individual images are to be displayed
        in this :py:class:`QMultiImageView`.
        """
        return self._imageSize

    def spacing(self) -> int:
        """The spacing to be put between the images displayed
        in this :py:class:`QMultiImageView`. At the outer boundary,
        half of the spacing will be used. That is, each grid cell
        occupies imageSize + spacing pixel, with the image centered
        in that cell.
        """
        return self._spacing

    def setImages(self, images: Iterable[Imagelike]) -> None:
        """Set the images to be displayed by this
        :py:class:`QMultiImageView`.

        Arguments
        ---------
        images:
            The images to be added. These may be given in any image
            format accepted by the toolbox, but will be converted
            to :py:class:`QImage` internally.
        """
        self._currentIndex = -1
        self._qimages = [imageToQImage(image) for image in images]
        self._gridChanged()

    def setImagesFromRegions(self, image: Imagelike,
                             regions: Iterable[Region]) -> None:
        """Set the images to be displayed from regions of a larger image.

        Arguments
        ---------
        image:
            The large image from which regions are cut out.
        regions:
            The regions to extract from that image. The regions
            may contain further annotations which may be used to
            influence the display.
        """
        if image is None:
            self._image = None
            self._regions = []
            self.setImages([])
        elif True:
            self._image = Image.as_array(image)
            self._regions = list(regions)
            self.setImages(region.extract_from_image(self._image)
                           for region in self._regions
                           if isinstance(region.location, BoundingBox))
        else:  # FIXME[todo]
            self._qimage = imageToQImage(image)
            self._regions = list(regions)
            self._rects = [self._rects_for_image(index)
                           for index in range(len(self._regions))]

    def _rects_for_image(self, index: int) -> Tuple[QRect, QRect]:
        """
        """
        bbox = self._rects[index]
        source = QRect(bbox.x, bbox.y, bbox.width, bbox.height)
        position = QPoint(self._spacing // 2, self._spacing // 2)
        if self.orientation() == Qt.Horizontal:
            position.setX(position.x() +
                          index * (self._imageSize.width() + self._spacing))
        else:
            position.setY(position.y() +
                          index * (self._imageSize.height() + self._spacing))

        target = QRect(position, self._imageSize)
        return source, target

    def currentImage(self) -> int:
        """The index of the currently selected image.
        """
        return self._currentIndex

    @pyqtSlot(int)
    def setCurrentImage(self, index: int) -> None:
        """Set the index of the currently selected image.
        """
        if index > -1:
            if not self.count():
                index = -1
            elif index >= self.count():
                index = -1
            elif index < 0:
                index = -1
        if index != self._currentIndex:
            self._currentIndex = index
            self.currentImageChanged.emit(index)
            self.update()

    def invalid(self, index: int = None) -> bool:
        """Check if an image in this :py:class:`QMultiImageView` is invalid.
        Images can have a flag, marking them as valid or invalid.

        Arguments
        ---------
        index:
            An index identifying the image to be checked. If no index is
            provided, the currently selected image is used. If no index
            can be determined, the method will return `False`.
        """
        if index is None:
            index = self.currentImage()
        if not 0 <= index < len(self._qimages) or not self._regions:
            return False  # no index could be determined

        region = self._regions[index]
        return getattr(region, 'invalid', False)

    def invalidate(self, index: int = None) -> None:
        """Invalidate an image in this :py:class:`QMultiImageView`.
        Invalid images will be displayed in a different way as
        valid images.

        Arguments
        ---------
        index:
            An index identifying the image to be invalidated. If no index is
            provided, the currently selected image is used. If there is
            no such index, the method will do nothing.
        """
        if index is None:
            index = self.currentImage()
        if not self._qimages or not 0 <= index < len(self._qimages):
            return

        region = self._regions[index]
        region.invalid = not getattr(region, 'invalid', False)
        self.annotationsChanged.emit(index)
        self.update()

    def imagePosition(self, index: int = None) -> QPoint:
        """Return the center point of the image with the given index.

        Arguments
        ---------
        index:
            An index identifying the desired image. If no index is
            provided, the currently selected image is used.

        Result
        ------
        center:
            The center point of the given image or `None` if no valid
            image was specified.
        """
        if index is None:
            index = self.currentImage()
        if not 0 <= index < len(self._qimages):
            return None

        horizontal_skip = self._spacing + self._imageSize.width()
        vertical_skip = self._spacing + self._imageSize.height()
        row, column = self._gridPosition()

        return QPoint(column * horizontal_skip + (horizontal_skip // 2),
                      row * vertical_skip + (horizontal_skip // 2))

    @protect
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """A mouse press selects the current image.
        """
        row = event.y() // (self._imageSize.width() + self._spacing)
        column = event.x() // (self._imageSize.width() + self._spacing)
        self._setGridPosition(row, column)

    @protect
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """A mouse double click invalidates the current image.
        """
        self.invalidate()

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process `QKeyEvent`s. Use cursor keys to change the currently
        selected image.

        Arguments
        ---------
        event:
            The key event to process.
        """
        key = event.key()

        if key == Qt.Key_Space:
            self.setCurrentImage(None)
        elif key in (Qt.Key_Left, Qt.Key_Up, Qt.Key_Right, Qt.Key_Down):
            row, column = self._gridPosition()
            if row < 0:
                row, column = (0, 0) if self.count() else (-1, -1)
            elif key == Qt.Key_Left:
                column = max(column-1, 0)
            elif key == Qt.Key_Right:
                column = column+1 % self.columns()
            elif key == Qt.Key_Up:
                row = max(row-1, 0)
            elif key == Qt.Key_Down:
                row = row+1 % self.rows()
            self._setGridPosition(row, column)
        elif key in (Qt.Key_Plus, Qt.Key_Minus):
            grid = list(self._grid)
            if grid[0] is None:
                if key == Qt.Key_Plus:
                    grid[1] += 1
                elif grid[1] > 1:
                    grid[1] -= 1
            elif grid[1] is None:
                if key == Qt.Key_Plus:
                    grid[0] += 1
                elif grid[0] > 1:
                    grid[0] -= 1
            self.setGrid(tuple(grid))
        elif key == Qt.Key_Delete:
            self.invalidate()
        elif key == Qt.Key_D:
            self._debug()
        else:
            super().keyPressEvent(event)

    @protect
    def paintEvent(self, event: QPaintEvent) -> None:
        """Process the paint event by repainting this Widget.

        Parameters
        ----------
        event:
            The :py:class:`QPaintEvent`paint eve
        """
        painter = QPainter()
        painter.begin(self)

        line_width = 4
        color = Qt.blue
        bluePen = QPen(color)
        bluePen.setWidth(line_width)
        redPen = QPen(Qt.red)
        redPen.setWidth(2 * line_width)

        horizontal_skip = self._spacing + self._imageSize.width()
        vertical_skip = self._spacing + self._imageSize.height()
        half_spacing = self._spacing // 2
        position = QPoint(half_spacing, half_spacing)
        for index, qimage in enumerate(self._qimages):
            # update position
            row, column = self._gridPosition(index)
            position = QPoint(column * horizontal_skip + half_spacing,
                              row * vertical_skip + half_spacing)
            rect = QRect(position, self._imageSize)
            painter.drawImage(rect, qimage)

            # draw decoration around current image
            if index == self._currentIndex:
                painter.setPen(bluePen)
                painter.drawRect(rect.x()-2, rect.y()-2,
                                 rect.width()+4, rect.height()+4)

            # draw specific decorations
            if self.invalid(index):
                painter.setPen(redPen)
                painter.drawLine(rect.topLeft(), rect.bottomRight())
                painter.drawLine(rect.topRight(), rect.bottomLeft())

        painter.end()

    def _debug(self) -> None:
        """Output some debug information to standard output.
        """
        print(f"QMultiImageView: {self.count()} images, grid={self.grid()},"
              f"size={self.size()}, sizeHint={self.sizeHint()}")
