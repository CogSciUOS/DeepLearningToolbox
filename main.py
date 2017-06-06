import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.models import Model
import keras
import math

import random
#k=np.load('data.npy')

plt.ion()
#plt.imshow(k[0,:,:,0])
#model = load_model('my2c2d_model.h5')

def preparedata(modelfilename,datafilename):
    '''loads the model and the data'''
    return (load_model(modelfilename),np.load(datafilename))



def activations(model,layername,inputdata,dataindex):

    #model = load_model(name)
    intermediate_layer_model = Model(input=model.input, output=model.get_layer(layername).output)
    intermediate_output = intermediate_layer_model.predict(inputdata[dataindex:dataindex+1,:,:,0:1])
    return intermediate_output
class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Activations'
        self.width = 1800
        self.height = 900
        self.sample_index=0
        self.initUI()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)


        for i in range(len(model.get_config())):
            btn = QPushButton(model.get_config()[i]['config']['name'], self)
            btn.resize(btn.sizeHint())
            btn.move(1500, 120+100*i)
            btn.clicked.connect(self.layerbuttonClicked)


        btnnext=    QPushButton('next', self)
        btnnext.resize(btn.sizeHint())
        btnnext.move(1500, 120+100*7)
        btnnext.clicked.connect(self.nextsamplebuttonclickked)

        self.show()
    def nextsamplebuttonclickked(self):
        self.sample_index+=1
        print(self.sample_index)

    def layerbuttonClicked(self,sample_index):
        sender = self.sender()
        m = PlotCanvas(self, width=9, height=9,layer=sender.text(),sample_index=self.sample_index)
        print(self.sample_index)
        m.move(0,0)
        m.show()
        print(sender.text())
        im=plotimage(self, width=4, height=4,sample_index=self.sample_index)
        im.move(900,0)
        im.show()

class plotimage(FigureCanvas):
    def __init__(self, parent=None, width=9, height=9, dpi=100,sample_index=None):
        fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = fig.add_subplot(111)
        self.axes.axis('off')

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plotinput(sample_index)

    def plotinput(self,dataindex):
        ax1 = self.figure.add_subplot(111)
        ax1.axis('off')
        ax1.imshow(k[dataindex,:,:,0],cmap='gray')

        #ax.set_title('PyQt Matplotlib Example')
        self.draw()





class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=9, height=9,layer=None, dpi=100,sample_index=None):

        fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = fig.add_subplot(111)
        self.axes.axis('off')

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plotactivat(layer,sample_index)






    def plotactivat(self,layer,dataindex):

        #intermediate_layer_model = Model(input=model.input, output=model.get_layer('convolution2d_4').output)

        intermediate_output = activations(model,layer,k,dataindex)
        #intermediate_output=intermediate_output.squeeze()
        vm=None
        if (len(intermediate_output.shape)==2):
            intermediate_output=intermediate_output.reshape(1,1,1,intermediate_output.shape[1])
            vm=np.max(intermediate_output)

        '''
        result=np.concatenate()
        '''
        nbofplots=intermediate_output.shape[3]
        imagesize=intermediate_output.shape[1]
        '''
        for i in range(nbofplots):
                    ax = self.figure.add_subplot(int(np.sqrt(nbofplots))+1,int(np.sqrt(nbofplots))+1,i+1)
                    ax.axis('off')

                    ax.imshow(intermediate_output[0,:,:,i],vmin=0,vmax=vm,cmap='gray')
        '''

        ncolumns=nraws=math.ceil(np.sqrt(nbofplots))

        ishow=np.zeros([imagesize*ncolumns,imagesize*ncolumns])
        intermediate_output=np.swapaxes(intermediate_output,0,3)
        print([ncolumns,nbofplots])
        for i in range(ncolumns):
            print([i,imagesize*(nbofplots-i*ncolumns)])
            ishow[i*imagesize:(i+1)*imagesize,0:imagesize*(nbofplots-i*ncolumns)]=np.hstack(intermediate_output[i*ncolumns:(i+1)*ncolumns,:,:,0])

        #c=np.vstack(ishow)
        ax = self.figure.add_subplot(111)
        ax.axis('off')
        ax.imshow(ishow[:,:],vmin=0,vmax=vm,cmap='gray')

        #ax.set_title('PyQt Matplotlib Example')
        self.draw()

if __name__ == '__main__':

    (model,k)=preparedata('my2c2d_model.h5','data.npy')
    #sample_index=0
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
