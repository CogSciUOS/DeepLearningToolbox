
import math
import numpy as np

from PyQt5.QtCore import Qt, QPoint, QPointF, QSize, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen
from PyQt5.QtWidgets import QWidget, QToolTip


# FIXME[todo]
#  * analyse the possible data types of the matrices that can be displayed
#    and adapt the code accordingly.
#  * assign a more reasonable initial size when the QMatrixView is
#    embedded in a QScrollArea


class QMatrixView(QWidget):
    '''An experimental class to display a matrix (e.g. a correlation
    matrix between two networks).

        - The MatrixView allows to select an individual entry
          by a mouse click.
        - Once an entry is selected, it can be moved by the using
          the keyboard.
        - A selection is indicated by the "selected()" signal.
        - The MatrixView allows to specify a region so that only
          a submatrix gets displayed. This is intended for interaction
          with some zoom element.

    Implementation
    ---------------
    Zoom: we have decided to include the zoom functionality into this widget.
    That is, on every change of zoom parameters (zoom factor or offset)
    we compute an image depicting the selected region and display it
    in the image.
    The alternative would have been to use a more modular approach:
    just resize this widget uppon zooming and implement the zoom
    functionality into some other widget, e.g. some scrollbar widget.
    The reasons we did not follow this approach are mainly due to
    not knowing how we can zoom a widget in a sensible way:
    (1) resizing the widget seems to interpolate between pixels
    in images, but we want to have sharp pixels (corresponding to
    individual matrix entries) in an enlarged view.
    (2) Using a resized image for large matrices can require a very
    large image that has to be kept in memory and recomputed at every
    change of zoom level.

    We have choosen to represent the zoom data by a zoom factor and an
    offset (left upper corner of the visible part of the visible part
    of the matrix image). This representation prevents changing the
    aspect ratio of pixels (matrix entries), i.e. pixel will always
    appear as squares, not rectangles. The zoom factor describes the
    size of a single pixel (matrix entry): a zoom factor of 1.0 means
    one pixel per matrix entry, while with a zoom factor of 10.0 each
    entry is displayed as a 10*10 square.


    Signals
    -------
    selected
        The "selected" signal is emitted when a new entry is selected
        or when the current entry gets unselected.

    zoomed
        The zoom changed (zoom factor or zoom position).


    Attributes
    -----------
    matrix : numpy.ndarray
        The matrix to be displayed in this widget. None if no
        matrix is assigned.
    selectedPosition : (int,int)
        The indices (row and column) of the selected matrix entry.
        None if no element is selected.
    zoom : double
        The current zoom factor. The zoom factor is the number of pixels
        used to display a matrix entry.
    minZoom : double
        Minimal zoom factor
    maxZoom : double
        Maximal zoom factor
    offset : QPointF
        The offset of the viewable region of the zoomed image.
        We use a float representation here to be accurate during zooming.
    toolTipActive : bool
        A flag indicating whether tooltips shoould be shown.
    toolTipText : str
        The text to be displayed as tooltip.
    '''

    selected = pyqtSignal(object)

    zoomed = pyqtSignal(float)

    '''The Widget keeps its size on zoom. If the zoomed image is larger
    than the widget size, only a part will be displayed. This policy is
    useful if the widget is to be used stand alone.
    '''
    ZoomPolicyFixed = 0

    '''The Widget gets resized on zoom. The widget size will be adapted to
    the size of the zoomed matrix. This zoom policy is useful when
    embedding the widget into a QScrollArea.
    '''
    ZoomPolicyResize = 1

    '''The zoom policy to apply to this widget.'''
    zoomPolicy = 0

    def __init__(self, matrix, parent=None):
        '''Initialization of the QMatrixView.

        Parameters
        ----------
        matrix : numpy.ndarray
            The matrix to be displayed in this QMatrixView.
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)

        self.selectedPosition = None
        self.offset = QPointF(0, 0)
        self.zoom = 1.0
        self.setToolTip(False)
        self.setMatrix(matrix)

        self.initUI()

    def initUI(self):
        '''Initialize the user interface.
        '''

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by "Tab" key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)

    def setZoomPolicy(self, zoomPolicy):
        '''Set the zoom policy for this QMatrixView.  The zoom policy will
        control how the QMatrixView behaves on zooming.

        Parameters
        ----------
        zoomPolicy : QMatrixView.ZoomPolicy...
        '''
        self.zoomPolicy = zoomPolicy
        self.resetZoom()

    def setToolTip(self, active=True):
        '''Turn on/off the tooltips for this Widget.
        The tooltip will display the index and value for the matrix
        at the current mouse position.

        Parameters
        ----------
        active : bool or str
            Will turn the tooltips on or off. When a string is provided
            that string will be used as tooltip. The string can have
            up to three placeholders, being filled with 1=row, 2=column,
            3=matrix value.
        '''

        self.toolTipActive = active != False

        if isinstance(active, str):
            self.toolTipText = active
        elif active:
            self.toolTipText = "C({},{}) = {:.2f}"
        else:
            self.toolTipText = ""

        self.setMouseTracking(self.toolTipActive)

        if not self.toolTipActive:
            QToolTip.hideText()

    def setMatrix(self, matrix):
        '''Set the matrix to be displayed. Setting the matrix will
        reset the zoom and trigger a repaint.

        Parameters
        ----------
        matrix : numpy.ndarray
            The matrix to be displayed in this QMatrixView. Can
            be None to indicate that no data should be displayed.
        '''

        if matrix is not None:
            # FIXME: allow for different data formats or just provide
            # in the correct format!
            self.matrix = abs(matrix * 255).astype(np.uint8)

            self._image = QImage(self.matrix, *matrix.shape[::-1],
                                 QImage.Format_Grayscale8)
        else:
            self.matrix = None
            self._image = None

        self.resetZoom()
        self.setSelection(None)
        self.update()

    def setSelection(self, selection=None):
        '''Set the selected matrix entry. This will trigger a repaint
        to also display the selected entry.

        Parameters
        ----------
        selection : (int,int)
            The row and column of the selected entry. None can be
            be provided to cancel the selection.
        '''
        if self.matrix is None:
            selection = None
        if selection is not None:
            if selection[0] < 0:
                selection = (0, selection[1])
            elif selection[0] >= self.matrix.shape[0]:
                selection = (self.matrix.shape[0] - 1, selection[1])
            if selection[1] < 0:
                selection = (selection[0], 0)
            elif selection[1] >= self.matrix.shape[1]:
                selection = (selection[0], self.matrix.shape[1] - 1)

        if selection != self.selectedPosition:
            self.selectedPosition = selection
            self.selected.emit(self.selectedPosition)
            self.update()

    def zoomRatio(self):
        '''Get the zoom ratio. The zoom ratio is given by the size
        of the widget divided by the size of the zoomed image.
        '''
        return (self.width() / self._scaledImage.width(),
                self.height() / self._scaledImage.height())

    def setZoom(self, zoom=None):
        '''Set the zoom factor. The zoom factor specifies to be used to
        display a matrix entry (zoom = n means n*n pixels).

        Parameters
        ----------
        selection : float
            The new zoom factor. If the factor is out of range, the
            closest valid value is taken instead.
            be provided to cancel the selection.
        '''

        # check that the new zoom is in the valid range
        if zoom is not None:
            if zoom < self.minZoom:
                zoom = self.minZoom
            if zoom > self.maxZoom:
                zoom = self.maxZoom

        if zoom != self.zoom:
            self.zoom = zoom
            if (self.zoomPolicy == QMatrixView.ZoomPolicyResize and
                    zoom is not None and self._image is not None):
                self.resize(self._image.size() * self.zoom)
            self.updateZoom()

    def getOffset(self):
        '''Get the offset. The offset is the position of the (left upper
        corner of the) part of the zoomed image that is currently
        displayed.

        Returns
        -------
        QPointF: The offset of the left upper corner of the visible
            area of the zoomed matrix.

        '''
        return self.offset

    def setOffset(self, offset):
        '''Set the offset.
        '''
        x = max(0, offset.x())
        y = max(0, offset.y())
        if self._image:
            if self.zoom * self._image.width() - self.width() < x:
                x = self.zoom * self._image.width() - self.width()
            if self.zoom * self._image.height() - self.height() < y:
                y = self.zoom * self._image.height() - self.height()

        if x != self.offset.x() or y != self.offset.y():
            self.offset = QPointF(x, y)
            self.updateZoom()

    def resetZoom(self):
        '''Reset the zoom factor. This will trigger a repaint.
        '''
        if self.matrix is None:
            self.minZoom = self.maxZoom = 1.0
        elif self.zoomPolicy == QMatrixView.ZoomPolicyFixed:
            # minimum zoom: we want to fill the widget at least
            # in one dimension
            self.minZoom = min(self.width() / self.matrix.shape[0],
                               self.height() / self.matrix.shape[1])
            # maximum zoom: we want to have at least n entries
            # displayed in the widget
            n = 10
            self.maxZoom = min(self.width() / n, self.height() / n)
            self.maxZoom = max(self.minZoom, self.maxZoom)
        else:
            self.minZoom = 3.0
            self.maxZoom = 30.0

        self.setZoom(self.minZoom)

    def updateZoom(self):
        '''Update the zoomed image to be displayed in this widget.
        The image is generated from the matrix based on the
        current zomm and offset values.
        The updated version of the image is stored in the
        private field _ZoomedImage.
        '''

        # emit the "zoomed" signal and update the display
        self.zoomed.emit(self.zoom)
        self.update()

    def zoomedSize(self):
        '''Return the size of the zoomed image. If the zoom policy is set
        to ZoomPolicyResize, this is equivalent to self.size().

        Returns
        -------
        QSize : The size of the zoomed image, or None if no matrix is set.
        '''
        return self._image.size() * self.zoom if self._image is not None else self.size()

    def resizeEvent(self, event):
        '''Adapt to a change in size. The behavior dependes on the zoom
        policy.

        Parameters
        ----------
        event : QResizeEvent

        '''
        # This event handler is called after the Widget has been resized.
        # providing the new .size() and the old .oldSize().
        if self.zoomPolicy == QMatrixView.ZoomPolicyFixed:
            self.resetZoom()

    def paintEvent(self, event):
        '''Process the paint event by repainting this Widget.

        Parameters
        ----------
        event : QPaintEvent
        '''
        qp = QPainter()
        qp.begin(self)
        self._drawWidget(qp, event.rect())
        qp.end()

    def _drawWidget(self, qp, rect):
        '''Draw a given portion of this widget.
        Parameters
        ----------
        qp : QPainter
        rect : QRect
        '''

        # 1. draw the pixmap
        if self._image is not None:
            zoomedImage = self._createZoomedImage(rect)
            if zoomedImage:
                qp.drawImage(rect.topLeft(), zoomedImage)

        # 2. Insert the selection indicator
        if self.selectedPosition is not None:
            pen_width = 2

            pen = QPen(Qt.red)
            pen.setWidth(pen_width)
            qp.setPen(pen)
            p = self.positionOfEntry(*self.selectedPosition)
            qp.drawRect(math.floor(p.x()),
                        math.floor(p.y()),
                        math.ceil(self.zoom),
                        math.ceil(self.zoom))

    def _createZoomedImage(self, rect):
        '''Create a zoomed version of the matrix image.

        Parameters
        ----------
        rect : QRect
            A rectangle describing the visible region.

        Returns
        -------
        QImage : An image of the given size.
        '''
        if self._image is not None:
            # 0. get to top left corner
            if self.zoomPolicy == QMatrixView.ZoomPolicyFixed:
                topLeft = self.offset.toPoint()
            else:
                topLeft = rect.topLeft()

            # 1. get the relevant subimage
            x = math.floor(topLeft.x() / self.zoom)
            y = math.floor(topLeft.y() / self.zoom)
            w = min(math.ceil(1 + rect.width() / self.zoom),
                    self._image.width() - x)
            h = min(math.ceil(1 + rect.height() / self.zoom),
                    self._image.height() - y)
            smallImage = self._image.copy(x, y, w, h)

            # 2. Zoom the image to the desired size.
            #    We have to really zoom the Image, rather than
            #    just displaying a resized version of it,
            #    to get a "pixeled" (i.e. not interpolated) zoom
            imageSize = QSize(self.zoom * w, self.zoom * h)
            scaledImage = smallImage.scaled(imageSize)

            # 3. Cut out the desired part.
            #    As we want to allow for placing with subpixel
            #    accuracy, we may have to cut a small margin.
            suboffset = QPoint(math.floor(x * self.zoom),
                               math.floor(y * self.zoom))
            rect = QRect(topLeft - suboffset, rect.size())
            zoomedImage = scaledImage.copy(rect)

        return zoomedImage if self._image is not None else None

    def entryAtPosition(self, position):
        '''Compute the entry corresponding to some point in this widget.

        Parameters
        ----------
        position : QPoint
            The position of the point in question (in Widget coordinates).

        Returns
        -------
        pair of ints or None
            (i,j) the indices of the entry at the given position,
            or None if no entry corresponds to that position.
        '''

        if self.matrix is None:
            return None
        p = (self.offset + position) / self.zoom
        x = math.floor(p.x())
        y = math.floor(p.y())
        return (y, x) if ((x < self.matrix.shape[1]) and
                          (y < self.matrix.shape[0])) else None

    def positionOfEntry(self, row, col):
        '''Compute the position of a given entry in this widget.

        Parameters
        ----------
        row : int
            The row of the entry.
        col : int
            The column of the entry.

        Returns
        -------
        QPoint : The left upper corner of the region in this widget, at
            which the entry with the given row and column is displayed.
            The complete entry occupies a space of size zoom*zoom.
        '''
        return QPointF(col, row) * self.zoom - self.offset

    def minimumSizeHint(self):
        '''The minimum size hint. We compute the size hint by specifying a
        minimal zoom factor and a minimal number of entries to be
        displayed.

        Returns
        -------
        QSize : The minimal size of this QMatrixView.
        '''
        minEntries = 10
        return QSize(minEntries * self.minZoom, minEntries * self.minZoom)

    def keyPressEvent(self, event):
        '''Process special keys for this widgets.
        Allow moving selected entry using the cursor key.
        Allow to move the visible

        Parameters
        ----------
        event : QKeyEvent
        '''
        key = event.key()
        if event.modifiers() & Qt.ControlModifier:
            if key == Qt.Key_Plus:
                self.setZoom(1.01 * self.zoom)
                self.setOffset(self.offset)
            elif key == Qt.Key_Minus:
                self.setZoom(0.99 * self.zoom)
                self.setOffset(self.offset)
            elif key == Qt.Key_Left:
                self.setOffset(QPointF(self.offset.x() - self.zoom,
                                       self.offset.y()))
            elif key == Qt.Key_Up:
                self.setOffset(QPointF(self.offset.x(),
                                       self.offset.y() - self.zoom))
            elif key == Qt.Key_Right:
                self.setOffset(QPointF(self.offset.x() + self.zoom,
                                       self.offset.y()))
            elif key == Qt.Key_Down:
                self.setOffset(QPointF(self.offset.x(),
                                       self.offset.y() + self.zoom))
            else:
                event.ignore()
        else:
            # Space will toggle display of tooltips
            if key == Qt.Key_Space:
                self.setToolTip(not self.toolTipActive)
            # Arrow keyes will move the selected entry
            elif self.selectedPosition is not None:
                row, col = self.selectedPosition
                if key == Qt.Key_Left:
                    self.setSelection((row, col - 1))
                elif key == Qt.Key_Up:
                    self.setSelection((row - 1, col))
                elif key == Qt.Key_Right:
                    self.setSelection((row, col + 1))
                elif key == Qt.Key_Down:
                    self.setSelection((row + 1, col))
                else:
                    event.ignore()
            else:
                event.ignore()

    def mousePressEvent(self, event):
        '''Process mouse event. As we implement .mouseDoubleClickEvent(), we
        also provide stubs for the other mouse events to not confuse
        other widgets.

        Parameters
        ----------
        event : QMouseEvent
        '''
        pass

    def mouseReleaseEvent(self, event):
        '''Process mouse event. As we implement .mouseDoubleClickEvent(), we
        also provide stubs for the other mouse events to not confuse
        other widgets.

        Parameters
        ----------
        event : QMouseEvent
        '''
        pass

    def mouseDoubleClickEvent(self, event):
        '''Process a double click. We use double click to select a
        matrix entry.

        Parameters
        ----------
        event : QMouseEvent
        '''
        self.setSelection(self.entryAtPosition(event.pos()))

    def wheelEvent(self, event):
        '''Process the wheel event. The mouse wheel can be used for
        zooming.

        Parameters
        ----------
        event : QWheelEvent
            The event providing the angle delta.
        '''
        delta = event.angleDelta().y() / 120  # will be +/- 1
        position = (self.offset + event.pos()) / self.zoom

        zoom = (1 + delta * 0.01) * self.zoom
        self.setZoom(zoom)

        offset = (position * self.zoom) - event.pos()
        self.setOffset(offset)

        # We will accept the event, to prevent interference
        # with the QScrollArea.
        event.accept()

    # Attention: The mouseMoveEvent() is only called for regular mouse
    # movements, if mouse tracking is explicitly enabled for this
    # widget by calling self.setMouseTracking(True).  Otherwise it may
    # be called on dragging.
    def mouseMoveEvent(self, event):
        '''Process mouse movements.  If tooltips are active, information on
        the entry at the current mouse position are displayed.

        Parameters
        ----------
        event : QMouseEvent
            The mouse event, providing local and global coordinates.
        '''
        if self.toolTipActive:
            index = self.entryAtPosition(event.pos())
            if index is not None:
                row, col = index
                c = self.matrix[row, col]
                text = self.toolTipText.format(row, col, c)
                QToolTip.showText(event.globalPos(), text)

    def leaveEvent(self, event):
        '''Handle the mouse leave event.
        When tooltips are shown in this widgets, they should
        be removed when the mouse leaves the widget.

        Parameters
        ----------
        event : QEvent
        '''
        if self.toolTipActive:
            QToolTip.hideText()
