from os import listdir
from os.path import isfile, join
from random import randint

import numpy as np
from scipy.misc import imread

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import QWidget, QPushButton, QRadioButton, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QFileDialog

# FIXME[todo]: add docstrings!


class QInputSelector(QWidget):
    '''A Widget to select input data (probably images).  There are
    different modes of selection: from an array, from a file, from a
    directory or from some predefined dataset.
    '''

    
    '''The current mode: can be 'array' or 'dir'
    '''
    _mode : str = None

    '''An array of input data. Can be None.
    '''
    _dataArray : np.ndarray = None

    
    '''A directory containing input data files. Can be None.
    '''
    _dataDir : str = None

    '''A list of filenames in the dataDir. Will be None if no
    directory was selected. An empty list indicates that no
    suitable files where found in the directory.
    '''
    _dataDirFilenames : list = None


    '''The number of data entries in the current data source.
    '''
    numberOfElements : int = None

    '''The index of the current 
    '''
    index : int = None



    '''A signal emitted when new input data are selected.
    The signal will carry the new data and some text explaining
    the data origin. (np.ndarray, str)
    '''
    selected = pyqtSignal(object, str)

    
    def __init__(self, number : int = None, parent = None):
        '''Initialization of the QNetworkView.

        Arguments
        ---------
        parent : QWidget
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)

        self.initUI()


    def initUI(self):
        '''Initialize the user interface.
        '''

        self.prevButton = QPushButton('previous')
        self.prevButton.clicked.connect(self.buttonClicked)

        self.infoLabel = QLabel()
        self.infoLabel.setMinimumWidth(QFontMetrics(self.font()).width("8")*12)

        self.nextButton = QPushButton('next')
        self.nextButton.clicked.connect(self.buttonClicked)

        self.randomButton = QPushButton('random')
        self.randomButton.clicked.connect(self.buttonClicked)

        self.arrayButton = QRadioButton("Array")
        self.arrayButton.setChecked(True)
        self.arrayButton.toggled.connect(lambda:self._setMode('array'))
		
        self.dirButton = QRadioButton("Directory")
        self.dirButton.toggled.connect(lambda:self._setMode('dir'))

        self.openButton = QPushButton('Open...')
        self.openButton.clicked.connect(self.openButtonClicked)

        layout = QHBoxLayout()
        layout.addWidget(self.prevButton)
        layout.addWidget(self.infoLabel)
        layout.addWidget(self.nextButton)
        layout.addWidget(self.randomButton)
        layout.addWidget(self.arrayButton)
        layout.addWidget(self.dirButton)
        layout.addWidget(self.openButton)
        self.setLayout(layout)

    def _setMode(self, mode : str):
        if self._mode != mode:
            self._mode = mode

            if mode == 'dir':
                self.setNumberOfElements(None if self._dataDirFilenames is None else len(self._dataDirFilenames))
                self.dirButton.setChecked(True)
            else:
                self.setNumberOfElements(None if self._dataArray is None else len(self._dataArray))
                self.arrayButton.setChecked(True)

    def setNumberOfElements(self, number : int = None):
        if number is None or number <= 1:
            self.numberOfElements = None
        else:
            self.numberOfElements = number
        valid = self.numberOfElements is not None
        self.prevButton.setEnabled(valid)
        self.nextButton.setEnabled(valid)
        self.setIndex(0 if valid else None)

                
    def setDataArray(self, data : np.ndarray = None):
        '''Set the data array to be used. 

        Arguments
        ---------
        data:
            An array of data. The first axis is used to select the
            data record, the other axes belong to the actual data.
        '''

        if data.ndim < 4:
            print("OLD data shape: {}".format(data.shape))
            data = data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
            print("NEW data shape: {}".format(data.shape))
        self._dataArray = data
        self._setMode('array')


    def setDataFile(self, filename : str):
        '''Set the data file to be used. 
        '''
        data = np.load(filename, mmap_mode = 'r')
        self.setDataArray(data)


    def setDataDirectory(self, dirname : str = None):
        '''Set the directory to be used for loading data. 
        '''
        self._dataDir = dirname
        if self._dataDir is None:
            self._dataDirFilenames = None
            self.setNumberOfElements(None)
        else:
            self._dataDirFilenames = [f for f in listdir(self._dataDir)
                                      if isfile(join(self._dataDir, f))]

            self.setNumberOfElements(len(self._dataDirFilenames))
        self.dirButton.setChecked(True)
 
        self._setMode('dir')

    def setDataSet(self, name : str):
        '''Set a data set to be used.

        Arguments
        ---------
        name:
            The name of the dataset. The only dataset supported up to now
            is "mnist".
        '''
        if name == 'mnist':
            from keras.datasets import mnist
            data = mnist.load_data()[0][0]
            self.setDataArray(data)
        else:
            raise ValueError("Unknown dataset: {}".format(name))


    def selectIndex(index):
        if self.numberOfElements:
            self.index = (index % self.numberOfElements)
        else:
            self.index = None
        self.update();        




    def buttonClicked(self):
        '''Callback for clicking the "next" and "prev" sample button.
        '''

        if self.index is None:
            index = None
        elif self.sender() == self.prevButton:
            index = self.index - 1
        elif self.sender() == self.nextButton:
            index = self.index + 1
        elif self.sender() == self.randomButton:
            index = randint(0, self.numberOfElements)
        else:
            index = None
        self.setIndex(index)


    def setIndex(self, index = None):
        if self.index != index:           
            if index is None or self.numberOfElements is None:
                self.index = None
            elif index < 0:
                self.index = 0
            elif index >= self.numberOfElements:
                self.index = self.numberOfElements - 1
            else:
                self.index = index
                
            if self.index is None:
                data = None
                info = "None"
            elif self._mode == 'array':
                data = self._dataArray[self.index:self.index+1,:,:,0:1]
                info = str(self.index) + "/" + str(self.numberOfElements)
            elif self._mode == 'dir':
                data = imread(join(self._dataDir,
                                   self._dataDirFilenames[self.index]))
                info = self._dataDirFilenames[self.index]
            else:
                data = None
                info = self._mode

            self.infoLabel.setText(info)
            # FIXME[bug]: there is an error in PyQt forbidding to emit None
            # signals.
            if data is None: data = np.ndarray(())
            self.selected.emit(data, info)


    
    def openButtonClicked(self):
        if self._mode == 'array':
            # FIXME[todo]:
            pass
        elif self._mode == 'dir':
            directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            self.setDataDirectory(directory)




class QInputInfoBox(QLabel):
    
    def __init__(self, parent = None):
        super().__init__(parent)
        self.showInfo()

    def showInfo(self, data = None, description = None):
        input_text = "<b>Input image:</b><br>\n"
        input_text += "Description: {}<br>\n".format(description)
        if data is not None:
            input_text += "Input shape: {} ({})<br>\n".format(data.shape, data.dtype)
        self.setText(input_text)

