import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QTabWidget


from qtgui.matrixview import MatrixView


# There are three classes defined in this file:
#   App:         the main window of the application
#   plotimage:   a Canvas to draw the input figure
#   PlotCanvas:  a Canvas to draw the activation

class App(QMainWindow):

    def __init__(self, network=None, data=None):
        super().__init__()

        # prepare matplotlib for interactive plotting on the screen
        plt.ion()

        self.network = network
        self.data = data

        layers = self.network.get_layer_list()
        self.layer_label = layers[0]
        self.sample_index=0
        self.initUI()
        self.update(input=True)


    def initUI(self):
        '''Initialize the graphical components of this user interface.
        '''

        self.title = 'Activations'

        self.left = 10
        self.top = 10
        self.width = 1800
        self.height = 900

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
	
        main = QWidget()
        experiments = QWidget()

        self.initMain(main)
        self.initExperiments(experiments)

        self.tabs = QTabWidget(self);
        self.tabs.setFixedSize(self.width-5, self.height-5)
        self.tabs.addTab(main,"Main")
        self.tabs.addTab(experiments,"Experiments")

        #window->setCentralWidget(centralWidget)
        self.show()


    def initMain(self, container):

        # a column of buttons to select the network layer
        self.old_sender = None
        for i,name in enumerate(self.network.get_layer_list()):
            btn = QPushButton(name, container)
            btn.resize(btn.sizeHint())
            btn.move(1500, 120+100*i)
            btn.clicked.connect(self.layerbuttonClicked)
            # FIXME[hack]: we need a better concept to access buttons!
            if self.layer_label == name:
                btn.setStyleSheet("background-color: red")
                self.old_sender = btn

        # the "next" button: used to load the next image 
        btnnext = QPushButton('next', container)
        btnnext.resize(btn.sizeHint())
        btnnext.move(1500, 120+100*7)
        btnnext.clicked.connect(self.nextsamplebuttonclickked)

        # a small textfield to display the name of the current image
        self.img_counter = QLabel("0", container)
        self.img_counter.move(1500, 80+100*7)

        # canvas_input: a canvas to display the input image
        self.canvas_input = MyImage(container, width=4, height=4)
        self.canvas_input.move(900,0)

        # canvas_activation: a canvas to display a layer activation
        self.canvas_activation = PlotCanvas(container, width=9, height=9)
        self.canvas_activation.move(0,0)

        # canvas_input2: a canvas to display the input image
        # (mayb be more efficient - check!)
        self.canvas_input2 = MyImage2(container)
        self.canvas_input2.move(900,400)
        self.canvas_input2.resize(300,300)

        # info_box: display input and layer info
        self.info_box = InfoBox(container)
        self.info_box.move(900,700)
        self.info_box.resize(300,200)


    def initExperiments(self, container):

        # Correlation matrix view
        correlations = np.random.rand(64,64) * 2 - 1
        self.matrix_view = MatrixView(correlations, container)
        self.matrix_view.move(10,10)
        self.matrix_view.resize(500,500)

        self.matrix_view.update()

    def nextsamplebuttonclickked(self):
        '''Callback for clicking the "next" sample button.
        '''
        self.sample_index = (self.sample_index + 1) % len(self.data)
        print("nextsamplebuttonclickked: {}".format(self.sample_index))
        self.update(input=True)


    def layerbuttonClicked(self, sample_index):
        '''Callback for clicking one of the layer buttons.
        '''
        sender = self.sender()  
        self.layer_label = sender.text()
        print("layerbuttonClicked: {}".format(self.layer_label))
        self.update(activation=True)

        # FIXME[hack]: we need a better concept to access buttons!
        sender.setStyleSheet("background-color: red")
        if self.old_sender:
            self.old_sender.setStyleSheet("")
        self.old_sender = sender


    def update(self, input=False, activation=None):
        '''Update the interface. This method should be called whenever
        the state of the application was changed.
        '''
        print("update: {} ({}) ".format(self.layer_label,self.sample_index))

        if input:
            self.canvas_input.myplot(self.data[self.sample_index,:,:,0])
            self.canvas_input2.myplot(self.data[self.sample_index,:,:,0])
            self.info_box.showInputInfo("{}/{}".format(self.sample_index,len(self.data)), self.data.shape[1:3])
            self.img_counter.setText("{}/{}".format(self.sample_index,len(self.data)))
            if activation is None: activation = True 


        if activation:
            activations = self.network.get_activations(self.layer_label,
                                                       self.data[self.sample_index:self.sample_index+1,:,:,0:1])
            self.canvas_activation.plotactivat(activations)
            self.info_box.showLayerInfo(activations.shape,
                                        self.network.get_layer_info(self.layer_label))
        



class MyImage(FigureCanvas):
    '''A simple class to display an image, using a MatPlotLib figure.
    '''

    def __init__(self, parent=None, width=9, height=9, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.axis('off')

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.draw()


    def myplot(self, image):
        '''Plot the given image.
        '''
        self.axes.imshow(image,cmap='gray')
        self.draw()





class PlotCanvas(FigureCanvas):
    '''A special kind of matplotlib backend that is used to plot the
    activation.

    '''

    def __init__(self, parent=None, width=9, height=9, dpi=100):

        fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = fig.add_subplot(111)
        self.axes.axis('off')

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.draw()


    def plotactivat(self, intermediate_output):
        '''Plot the activation patterns for a complete layer.
        '''

        print("plotactivat: intermediate_output.shape = {}".format(intermediate_output.shape))
        fully_connected = (len(intermediate_output.shape)==2)

        if fully_connected:
            intermediate_output = intermediate_output.reshape(1,1,1,intermediate_output.shape[1])
        # the axis of intermediate_output are:
        # (batch_size, width, height, output_channels)
            
        vm = np.max(intermediate_output) if fully_connected else None

        # number of plots: plot all output channels
        nbofplots = intermediate_output.shape[3]

        # image size: filter size (or a single pixel per neuron)
        imagesize = intermediate_output.shape[1]

        # number of columns and rows (arrange as square)
        ncolumns = nraws = math.ceil(np.sqrt(nbofplots))

        # the pixel map to be shown
        ishow = np.zeros([imagesize*ncolumns,imagesize*ncolumns])
        
        intermediate_output = np.swapaxes(intermediate_output,0,3)
        # the axis of intermediate_output are:
        # (output_channels, width, height, batch_size)
        
        print("plotactivat: columns = {}, plots = {}".format(ncolumns,nbofplots))
        for i in range(ncolumns):
            ishow[i*imagesize:(i+1)*imagesize,0:imagesize*(nbofplots-i*ncolumns)]=np.hstack(intermediate_output[i*ncolumns:(i+1)*ncolumns,:,:,0])

        # FIXME: no negative values?
        self.axes.imshow(ishow,vmin=0,vmax=vm,cmap='gray')
        self.draw()




from PyQt5.QtGui import QImage, QPixmap

class MyImage2(QLabel):
    '''An experimental class to display images using the QPixmap
    class.  This may be more efficient than using matplotlib for
    displaying images.
    '''

    def __init__(self, parent):
        super().__init__(parent)
        self.setScaledContents(True)
        # an alternative may be to call 
        #     pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        # in the myplot method.


    def myplot(self, image):
        '''Display the given image. Image is supposed to be a numpy array.
        '''

        print("myplot: {} ({}) {}-{}".format(image.shape,image.dtype,image.min(),image.max()))

        # To construct a 8-bit monochrome QImage, we need a uint8
        # numpy array
        if image.dtype != np.uint8: 
            image = (image*255).astype(np.uint8)
            
        qtimage = QImage(image, image.shape[1], image.shape[0],
                         QImage.Format_Grayscale8)
        pixmap = QPixmap(qtimage)
        self.setPixmap(pixmap)


class InfoBox(QLabel):

    def __init__(self, parent):
        super().__init__(parent)
        self.layer_text = "<b>Layer info:</b><br>\n"
        self.input_text = "<b>Input image:</b><br>\n"

    def showLayerInfo(self, shape, info):
        self.layer_text = "<b>Layer info:</b> {}<br>\n".format(info['name'])
        self.layer_text += ("Fully connected" if len(shape) == 2 else "Convolutional") + "<br>\n"
        self.layer_text += "Shape: {}<br>\n".format(shape)
        self.showInfo()

    def showInputInfo(self, name, shape):
        self.input_text = "<b>Input image:</b><br>\n"
        self.input_text += "Name: {}<br>\n".format(name)
        self.input_text += "Input shape: {}<br>\n".format(shape)
        self.showInfo()

    def showInfo(self):
        self.setText(self.input_text + "<br>\n<br>\n" +
                     self.layer_text)
        
