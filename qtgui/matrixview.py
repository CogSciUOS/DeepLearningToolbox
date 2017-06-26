from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QIcon


from PyQt5 import QtCore
from PyQt5.QtCore import QSize

from PyQt5.QtGui import QImage, QPixmap, QPalette

from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QScrollArea

import numpy as np

class MatrixView(QWidget):
    '''An experimental class to display the correlation matrix between two
    networks.
    '''

    def __init__(self, parent, correlations):
        super().__init__(parent)
        self.correlations = correlations
        
        self.zoom = 100
        self.mousePosition = (0,0)

        self.initUI()

    def initUI(self):

        self.matrixViewImage = MatrixViewImage(self)
        self.zoomLabel = QLabel("Zoom")
        self.zoomLabel.mousePressEvent = self.zoomEvent
        self.mouseLabel = QLabel("Mouse")
        
        infoline = QHBoxLayout()
        infoline.addWidget(self.zoomLabel)
        infoline.addWidget(self.mouseLabel)

        layout = QVBoxLayout()
        layout.addWidget(self.matrixViewImage)
        layout.addLayout(infoline)
    
        box = QGroupBox("Correlation Matrix", self)
        box.setLayout(layout)
        box.resize(300,300)

        self.update()
        self.show()
    
        
    def getCorrelations(self):
        return self.correlations

    def update(self):
        self.matrixViewImage.update()
        self.updateZoom()
        self.updateMouse()

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
        
    def setMouse(self, mousePosition):
        self.mousePosition = mousePosition
        self.updateMouse()

    def updateMouse(self):
        self.mouseLabel.setText("Mouse: {}".format(self.mousePosition))



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
        scaledImage = self.qtImage.scaled(imageSize) #, Qt.KeepAspectRatio);
        pixmap = QPixmap(scaledImage)
        self.imageLabel.setPixmap(pixmap);
        self.imageLabel.resize(pixmap.size());
        
        self.adjustScrollBar(self.horizontalScrollBar(), scaleFactor) 
        self.adjustScrollBar(self.verticalScrollBar(), scaleFactor) 

    def adjustScrollBar(self, scrollBar, factor):
        print("adjustScrollBar: value: {}, step: {}".format(scrollBar.value(), scrollBar.pageStep()))
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep()/2)))

    def wheelEvent(self,event):
        self.controller.changeZoom(event.angleDelta().y()/120)

    def mousePressEvent(self, event):
        position = event.pos()
        print("pressed here: " + str(position.x()) + ", " + str(position.y()))
        print("size: {}x{} ({})".format(self.width(),self.height(), self.controller.getCorrelations().shape))
        x = int((position.x()+self.horizontalScrollBar().value())/self.controller.zoom * 100 / self.ratio)
        y = int((position.y()+self.verticalScrollBar().value())/self.controller.zoom * 100 / self.ratio)
        self.controller.setMouse((x,y))

    def mouseMoveEvent(self, event):
        position = event.pos()
        print("moved here: " + str(position.x()) + ", " + str(position.y()))

    def mouseReleaseEvent(self, event):
        position = event.pos()
        print("released here: " + str(position.x()) + ", " + str(position.y()))

