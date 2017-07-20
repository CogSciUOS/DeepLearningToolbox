from PyQt5.QtCore import Qt, QPoint, QPointF, QSize, QSizeF, QRect, QRectF, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter, QPen, QPaintEvent
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QWidget, QLabel, QGroupBox
from PyQt5.QtWidgets import QToolTip
from PyQt5.QtWidgets import QAbstractScrollArea, QScrollArea

import math
import numpy as np


# FIXME[todo]:
#  * position the scrollbar during zoom in / zoom out
#    so that the mouse pointer remains over the same pixel
#  * emit a signal uppon selection of a pair
#  * allow to set the selected pair from outside (without emitting a signal)
#  * remove the box.resize but make the whole QWidget to be resizable,
#    provide reasonable minimum size, ...
#  * rethink the possible actions:
#    1) set value(s) from outside
#       (a) for initialization (may emit signals)
#       (b) to react on change in other Widget (should not emit signal
#           as the other Widget is responsible for doing so)
#    2) there may by multiple way to change a value, e.g. changing
#       the correlation matrix should also unset the position.
#       Update of view should only occur after all changes are done
#       (to avoid multiple updates).
#    So it seems that when setting a value, multiple arguments have
#    to be provided:
#  * Refactor:
#    - make the matrix view (without zoom, box and labels),
#      but with the ability to select entries, available as a separate
#      component.
#      (probably the correlation matrix needs only to be stored here)
#    - make the scroll view (with zoom capability, but without label)
#      available as a separate component.
#    - Provide a container that encapsulates the zoomable matrix and labels.
#      The container should provide an interface to access the internal
#      state (selected entry, correlation value(s), signal, initialisation)

# from PyQt5.QtCore import pyqtSignal
#
# @pyqtSignal(object)
# def on_correlation_changed(self, pair)
        
class MatrixView(QWidget):
    '''An experimental class to display the correlation matrix between two
    networks.
    '''

    selected = pyqtSignal(object)

    def __init__(self, correlations, parent = None):
        super().__init__(parent)
        self.correlations = correlations
        
        self.zoom = 100
        self.selectedPosition = None
        self.zoomPosition = (0,0)
        
        self.initUI()

        #self.selected.connect(self.updateSelected) 


    def initUI(self):
        '''Initialize the user interface.
        '''

        self.matrixViewImage = MatrixViewImage(self)
        self.zoomLabel = QLabel("Zoom")
        self.zoomLabel.mousePressEvent = self.zoomEvent
        
        self.selectionLabel = QLabel("Selection")
        
        infoline = QHBoxLayout()
        infoline.addWidget(self.zoomLabel)
        infoline.addWidget(self.selectionLabel)

        layout = QVBoxLayout()
        layout.addWidget(self.matrixViewImage)
        layout.addLayout(infoline)
    
        box = QGroupBox("Correlation Matrix", self)
        box.setLayout(layout)
        box.resize(300,400)

        self.update()
        self.show()

        
    def getCorrelations(self):
        return self.correlations

    def getCorrelation(self, x, y):
        return self.correlations[y,x]

    def update(self):
        self.matrixViewImage.update()
        self.updateZoom()
        self.updateSelectionLabel()

    def changeZoom(self, delta):
        self.setZoom(self.zoom + delta)

    def setZoom(self, zoom):
        self.zoom = max(10,zoom)
        self.updateZoom()
        
    def getZoom(self):
        return self.zoom

    def zoomEvent(self, event):
        self.setZoom(100)

    def updateZoom(self):
        self.zoomLabel.setText("Zoom: {}%".format(self.zoom))
        self.matrixViewImage.updateZoom()
        
    def setSelectedPosition(self, position, emitSignal = False):
        '''Set the selected position.
        The internal field will be set and the display will be
        updated accordingly.
        
        Args:
            position (pair of int): index of the selected entry 
                in the correlation matrix. May be None to indicate
                that no entry should be selected.
            emitSignal (bool): a flag indicating if the "selected"
                signal should be emit. If True, the signal will
                get the position as argument.
        '''
        self.selectedPosition = position
        self.updateSelectionLabel()
        # Update the display of the matrix view to reflect
        # the selected entry.
        self.matrixViewImage.updateZoom()
        if (emitSignal):
            self.selected.emit(position)

    def updateSelectionLabel(self):
        '''Update the label displaying the selected entry.
        '''
        
        if (self.selectedPosition is None) or (self.correlations is None):
            text = ""
        else:
            x = self.selectedPosition[0]
            y = self.selectedPosition[1]
            c = self.getCorrelation(x,y)
            text = "C({},{}) = {:.2f}".format(x,y,c)
        self.selectionLabel.setText(text)

        

    #def updateSelected(self, position):
    #    self.selectionLabel.setText("Mouse: {}".format(position))
    #    self.matrixViewImage.updateZoom()


class MatrixViewImage(QScrollArea):

    
    def __init__(self, parent):
        super().__init__(parent)

        self.controller = parent

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        
        # We set imageLabel's size policy to ignored, making the users
        # able to scale the image to whatever size they want when the
        # Fit to Window option is turned on. Otherwise, the default
        # size polizy (preferred) will make scroll bars appear when the
        # scroll area becomes smaller than the label's minimum size hint.
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored,
                                      QSizePolicy.Ignored)

        # We ensure that the label will scale its contents to fill all
        # available space, to enable the image to scale properly when
        # zooming. If we omitted to set the imageLabel's
        # scaledContents property, zooming in would enlarge the
        # QLabel, but leave the pixmap at its original size, exposing
        # the QLabel's background.
        self.imageLabel.setScaledContents(True)

        self.setBackgroundRole(QPalette.Base)
        self.setWidget(self.imageLabel)
        self.setVisible(False)

        self.imageLabel.mouseMoveEvent = self.mouseMoveEvent
        self.imageLabel.setMouseTracking(True)

    def update(self):
        self.updateImage()

    def updateImage(self):
        image = abs(self.controller.getCorrelations()*255).astype(np.uint8)
            
        self.qtImage = QImage(image, image.shape[1], image.shape[0],
                              QImage.Format_Grayscale8)
        self.ratio = min((self.width()-2)/self.qtImage.width(),
                         (self.height()-2)/self.qtImage.height())
        self.setVisible(True)
        self.updateZoom()

    def resizeEvent(self, event):
        print("resize: {}".format(self.size()))
        self.updateImage()
        
    def updateZoom(self):
        scaleFactor = self.controller.zoom / 100 * self.ratio

        # Just resizing the image will cause pixel interpolation
        #self.imageLabel.resize(scaleFactor * self.imageLabel.pixmap().size())

        # Therefore we create a rezized image and set it as a new pixmap.
        origSize = self.controller.getCorrelations().shape
        imageSize = QSize(scaleFactor*origSize[0], scaleFactor*origSize[1])
        scaledImage = self.qtImage.scaled(imageSize).convertToFormat(QImage.Format_RGB32) #, Qt.KeepAspectRatio);

        
        # Insert the selection indicator
        if self.controller.selectedPosition is not None:
            x,y = [scaleFactor * _ for _ in self.controller.selectedPosition]
            pen_width = 3

            painter = QPainter()
            pen = QPen(Qt.red)
            pen.setWidth(pen_width)
            painter.begin(scaledImage)
            painter.setPen(pen)
            painter.drawRect(x-0.5*pen_width,
                             y-0.5*pen_width,
                             scaleFactor+pen_width,
                             scaleFactor+pen_width)
            painter.end()

        pixmap = QPixmap(scaledImage)


        self.imageLabel.setPixmap(pixmap);
        self.imageLabel.resize(pixmap.size());
        
        self.adjustScrollBar(self.horizontalScrollBar(), scaleFactor) 
        self.adjustScrollBar(self.verticalScrollBar(), scaleFactor) 


    def adjustScrollBar(self, scrollBar, factor):
        print("adjustScrollBar: value: {}, step: {}".format(scrollBar.value(), scrollBar.pageStep()))
        #scrollBar.setValue(int(factor * scrollBar.value()
        #                       + ((factor - 1) * scrollBar.pageStep()/2)))


    def wheelEvent(self,event):
        self.controller.changeZoom(event.angleDelta().y()/120)

    def mousePressEvent(self, event):
        position = event.pos()
        #position = event.pos()
        #print("pressed here: " + str(position.x()) + ", " + str(position.y()))
        #print("size: {}x{} ({})".format(self.width(),self.height(), self.controller.getCorrelations().shape))

    def mouseReleaseEvent(self, event):
        #print("released here: " + str(position.x()) + ", " + str(position.y()))
        self.controller.setSelectedPosition(self.indexForPosition(event.pos()))

    def mouseMoveEvent(self, event):
        index = self.indexForPosition(event.pos())
        if index is not None:
            x,y = index
            c = self.controller.getCorrelation(x,y)
            text = "C({},{}) = {:.2f}".format(x,y,c)
            QToolTip.showText(event.globalPos(), text)
            # In most scenarios you will have to change these for
            # the coordinate system you are working in.
            # self, rect() );
        # QWidget::mouseMoveEvent(event);  // Or whatever the base class is.

    def indexForPosition(self, position):
        x = int((position.x()+self.horizontalScrollBar().value())/self.controller.zoom * 100 / self.ratio)
        y = int((position.y()+self.verticalScrollBar().value())/self.controller.zoom * 100 / self.ratio)
        result = (x,y)
        correlations = self.controller.correlations
        if correlations is None:
            result = None
        elif ((x >= correlations.shape[1]) or
              (y >= correlations.shape[0])):
            result = None
        return result





# FIXME:
#  - signal interface for zoom

# FIXME[lookup]
#  - event management: when (and how) should I forward events
#  - resizing and placement
#     - how to specify minimum/preferred/maximum size?
#     - what does the resize event do?


# FIXME: rename to QZoomableMatrixView

class MatrixWidget(QWidget):
    '''An experimental class to display a matrix (e.g. a correlation
    matrix between two networks.
        - The MatrixView allows to select an individual entry
          by a mouse click.
        - Once an entry is selected, it can be moved by the using
          the keyboard.
        - A selection is indicated by the "slected()" signal.
        - The MatrixView allows to specify a region so that only
          a submatrix gets displayed. This is intended for interaction
          with some zoom element.

    Implementation
    --------------
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


    Attributes:
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

    # FIXME: what interface?
    zoomed = pyqtSignal()


    def __init__(self, matrix, parent = None):
        super().__init__(parent)

        self.selectedPosition = None

        self.minZoom = 1.0
        self.maxZoom = 1.0
        self.zoom = 0.0

        self.offset = QPointF(0,0)

        self.toolTipActive = False
        self.toolTipText = ""

        self.setMatrix(matrix)

        self.initUI()
        #self.setToolTip()


    def initUI(self):

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by "Tab" key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)

        # FIXME:
        # Without setting this, there occurs an error uppon
        # construction. It seems that a paint event is submitted
        # before the class a
        self._zoomedImage = None
    

    def setToolTip(self, active = True):
        '''Turn on/off the tooltips for this Widget.
        The tooltip will display the index and value for the matrix
        at the current mouse position.
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
        '''

        if matrix is not None:
            # FIXME: just provide in the correct format!
            self.matrix = abs(matrix*255).astype(np.uint8)

            self._image = QImage(self.matrix, *matrix.shape[::-1],
                                 QImage.Format_Grayscale8)
        else:
            self.matrix = None
            self._image = None

        self.resetZoom()
        self.setSelection(None)
        self.update()
        

    def setSelection(self, selection = None):
        '''Reset the selection factor. This will trigger a repaint.
        '''
        if (selection is not None) and ((self.matrix is None) or
            (selection[0] < 0) or (selection[0] >= self.matrix.shape[0]) or
            (selection[1] < 0) or (selection[1] >= self.matrix.shape[1])):
            selection = None
        
        if selection != self.selectedPosition:
            self.selectedPosition = selection
            self.selected.emit(self.selectedPosition)
            self.update()


    def zoomRatio(self):
        '''Get the zoom ratio. The zoom ratio is given by the size
        of the widget divided by the size of the zoomed image.
        '''
        return (self.width / self._scaledImage.width(),
                self.height() / self._scaledImage.height())


    def setZoom(self, zoom = None):
        '''Set the zoom factor.
        '''

        # check that the new zoom is in the valid range
        if zoom is not None:
            if zoom < self.minZoom:
                zoom = self.minZoom
            if zoom > self.maxZoom:
                zoom = self.maxZoom
                
        if zoom != self.zoom:
            self.zoom = zoom
            self.updateZoom()


    def getOffset(self):
        '''Get the zoom position. The zoom position is position
        of the (left upper corner of the) the part of the zoomed
        image that is displayed
        '''
        return self.offset


    def setOffset(self, offset):
        '''Set the zoom position.
        '''
        x = max(0, offset.x())
        y = max(0, offset.y())
        if self._image:
            if self.zoom * self._image.width() - self.width() < x:
                x = self.zoom * self._image.width() - self.width() 
            if self.zoom * self._image.height() - self.height() < y:
                y = self.zoom * self._image.height() - self.height()

        if x != self.offset.x() or y != self.offset.y():
            self.offset = QPointF(x,y)
            self.updateZoom()


    def resetZoom(self):
        '''Reset the zoom factor. This will trigger a repaint.
        '''
        if self.matrix is None:
            self.minZoom = self.maxZoom = 1.0
        else:
            # minimum zoom: we want to fill the widget at least
            # in one dimension
            self.minZoom = min(self.width()/self.matrix.shape[0],
                               self.height()/self.matrix.shape[1])
            # maximum zoom: we want to have at least n entries
            # displayed in the widget
            n = 10
            self.maxZoom = min(self.width()/n, self.height()/n)
            self.maxZoom = max(self.minZoom, self.maxZoom)

        self.setZoom(1.0)



    def updateZoom(self):
        '''Update the zoomed image to be displayed in this widget.
        The image is generated from the matrix based on the
        current zomm and offset values.
        The updated version of the image is stored in the
        private field _ZoomedImage.
        '''

        self._zoomedImage = None
        
        # emit the "zoomed" signal and update the display
        self.zoomed.emit()
        self.update()

    def createZoomedImage(self):

        if self._image is not None:
            # 1. get the relevant subimage
            x = math.floor(self.offset.x()/self.zoom)
            y = math.floor(self.offset.y()/self.zoom)
            w = min(math.ceil(1+self.width()/self.zoom),
                    self._image.width()-x)
            h = min(math.ceil(1+self.height()/self.zoom),
                    self._image.height()-y) 
            smallImage = self._image.copy(x,y,w,h)

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
            offset = QPoint(self.offset.x(), self.offset.y())
            rect = QRect(offset - suboffset,self.size())
            self._zoomedImage = scaledImage.copy(rect)
        else:
            self._zoomedImage = None


    def zoomedSize(self):
        return self._image.size() * self.zoom if self._image is not None else self.size()


    def paintEvent(self, e):
        '''Process the paint event by repainting this Widget.
        '''
        print("PaintEvent: size={}, zoom={}, zoomed size={}".format(self.size(),self.zoom,self.zoomedSize()))
        qp = QPainter()
        qp.begin(self)
        self.drawWidget(qp)
        qp.end()


    def drawWidget(self, qp):
        '''Draw this widget.
        '''

        # 1. draw the pixmap
        #target = QRectF(0,0, self.width(), self.height())
        #source = QRectF(self.offset, QSizeF(self.size()))
        if self._image and not self._zoomedImage:
            self.createZoomedImage()
        if self._zoomedImage:
            qp.drawImage(QPoint(0,0), self._zoomedImage)

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



    def entryAtPosition(self, position):
        '''Compute the entry corresponding to some point in this widget.

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
        return (y,x) if ((x<self.matrix.shape[1]) and
                         (y<self.matrix.shape[0])) else None


    def positionOfEntry(self, row, col):
        '''Compute the position of a given entry in this widget.

        Return: QPoint
        -------
            The left upper corner of the region in this widget, at
            which the entry with the given coordinates is displayed.
            The complete entry occupies a space of size zoom*zoom.
        '''
        return QPointF(col, row) * self.zoom - self.offset


    def minimumSizeHint(self):
        minEntries = 10
        minZoomFactor = 3
        return QSize(minEntries * minZoomFactor, minEntries * minZoomFactor)


    def keyPressEvent(self, event):
        '''Process special keys for this widgets.
        Allow moving selected entry using the cursor key.
        Allow to move the visible 
        
        '''
        key = event.key()
        if self.selectedPosition is not None:
            if event.modifiers() & Qt.ControlModifier:
                x = self.offset.x()
                y = self.offset.y()
                w = self.zoom * self.matrix.shape[1]
                h = self.zoom * self.matrix.shape[0]
                if key == Qt.Key_Left and x >= 0:
                    self.setOffset(QPointF(max(x-self.zoom,0),y))
                elif key == Qt.Key_Up and y >= 0:
                    self.setOffset(QPointF(x,max(y-self.zoom,0)))
                elif key == Qt.Key_Right and x+self.width() < w:
                    self.setOffset(QPointF(min(w-self.width(),x+self.zoom),y))
                elif key == Qt.Key_Down and y+self.height() < h:
                    self.setOffset(QPointF(x,min(h-self.height(),y+self.zoom)))
                else:
                    event.ignore()
            else:
                row,col = self.selectedPosition
                if key == Qt.Key_Left and col>0:
                    self.setSelection((row,col-1))
                elif key == Qt.Key_Up and row>0:
                    self.setSelection((row-1,col))
                elif key == Qt.Key_Right and col+1<self.matrix.shape[1]:
                    self.setSelection((row,col+1))
                elif key == Qt.Key_Down and row+1<self.matrix.shape[0]:
                    self.setSelection((row+1,col))
                else:
                    event.ignore()
        else:
            event.ignore()

        # Space will toggle display of tooltips
        if key == 32:
            self.setToolTip(not self.toolTipActive)
            event.accept()


    def mousePressEvent(self, event):
        pass


    def mouseReleaseEvent(self, event):
        pass

    def mouseDoubleClickEvent(self, event):
        self.setSelection(self.entryAtPosition(event.pos()))

    def wheelEvent(self, event):
        delta = event.angleDelta().y()/120 # will be +/- 1
        position = (self.offset + event.pos()) / self.zoom

        zoom = (1 + delta * 0.01) * self.zoom
        self.setZoom(zoom)

        offset = (position * self.zoom) - event.pos()
        self.setOffset(offset)


    def resizeEvent(self, event):
        # This event handler is called after the Widget has been resized.
        print("resizeEvent: current size: {}, old size: {}, new size: {}".format(self.size(),event.oldSize(),event.size()))
        #self.resetZoom()


    # Attention: The mouseMoveEvent() is only called for regular mouse
    # movements, if mouse tracking is explicitly enabled for this
    # widget by calling self.setMouseTracking(True).  Otherwise it may
    # be called on dragging.
    def mouseMoveEvent(self, event):

        if self.toolTipActive:
            index = self.entryAtPosition(event.pos())
        
            if index is not None:
                x,y = index
                c = self.matrix[y,x]
                text = self.toolTipText.format(x,y,c)
                QToolTip.showText(event.globalPos(), text)
        #FIXME: to do ...
        # QWidget::mouseMoveEvent(event);  // Or whatever the base class is.


    def leaveEvent(self, event):
        '''Handle the mouse leave event.
        When tooltips are shown in this widgets, they should
        be removed when the mouse leaves the widget.
        '''
        if self.toolTipActive:
            QToolTip.hideText()








# When inheriting QAbstractScrollArea, you need to do the following:
#  - Control the scroll bars by setting their range, value, page step,
#    and tracking their movements.
#  - Draw the contents of the area in the viewport according to the
#    values of the scroll bars.
#  - Handle events received by the viewport in viewportEvent() -
#    notably resize events.
#  - Use viewport->update() to update the contents of the viewport
#    instead of update() as all painting operations take place on the viewport.


# The scroll bars and viewport should be updated whenever the viewport
# receives a resize event or the size of the contents changes. The
# viewport also needs to be updated when the scroll bars values
# change. The initial values of the scroll bars are often set when the
# area receives new contents.


class QZoomableScrollArea(QAbstractScrollArea):

    def __init__(self, parent = None):
        super().__init__(parent)

        #QSize areaSize = viewport().size();
        #QSize widgetSize = widget.size();

        #verticalScrollBar()->setPageStep(areaSize.height());
        #horizontalScrollBar()->setPageStep(areaSize.width());
        #verticalScrollBar()->setRange(0, widgetSize.height() - areaSize.height());
        #horizontalScrollBar()->setRange(0, widgetSize.width() - areaSize.width());
        #updateWidgetPosition();


    def viewportEvent(self, event):
        print("viewportEvent: {}".format(event))
        if not isinstance(event, QPaintEvent):
            self.viewport().update()
        else:
            self.myupdate()
        return False


    def setupViewport(self, zoomableWidget):
        pass # FIXME: connect to signal

    def myupdate(self):
        hvalue = self.horizontalScrollBar().value();
        vvalue = self.verticalScrollBar().value();
        topLeft = self.viewport().rect().topLeft();
        print("hvalue = {}, vvalue = {}, topLeft = {}".format(hvalue,vvalue,topLeft))

        areaSize = self.viewport().zoomedSize();
        widgetSize = self.size();
        
        self.verticalScrollBar().setPageStep(areaSize.height());
        self.horizontalScrollBar().setPageStep(areaSize.width());
        self.verticalScrollBar().setRange(0, areaSize.height() - widgetSize.height());
        self.horizontalScrollBar().setRange(0, areaSize.width() - widgetSize.width());


    def updateWidgetPosition(self):
        zoom = self.viewport().getZoom()
        position = self.verticalScrollBar().value

        #widget.move(topLeft.x() - hvalue, topLeft.y() - vvalue);

