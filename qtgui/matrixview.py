from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter, QPen
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QWidget, QLabel, QGroupBox, QScrollArea
from PyQt5.QtWidgets import QToolTip

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
