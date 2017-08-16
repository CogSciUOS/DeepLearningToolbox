from random import randint

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout

# FIXME[todo]: add docstrings!


class QInputSelector(QWidget):

    selected = pyqtSignal(int)
   
    numberOfElements : int = None
    index : int = None

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
        self.setNumberOfElements(number)


    def initUI(self):

        self.prevButton = QPushButton('previous')
        self.prevButton.clicked.connect(self.buttonClicked)

        self.infoLabel = QLabel()
        self.infoLabel.setMinimumWidth(QFontMetrics(self.font()).width("8")*12)

        self.nextButton = QPushButton('next')
        self.nextButton.clicked.connect(self.buttonClicked)

        self.randomButton = QPushButton('random')
        self.randomButton.clicked.connect(self.buttonClicked)


        layout = QHBoxLayout()
        layout.addWidget(self.prevButton)
        layout.addWidget(self.infoLabel)
        layout.addWidget(self.nextButton)
        layout.addWidget(self.randomButton)
        self.setLayout(layout)


    def setNumberOfElements(self, number : int = None):
        if number is None or number <= 1:
            self.numberOfElements = None
        else:
            self.numberOfElements = number
        valid = self.numberOfElements is not None
        self.prevButton.setEnabled(valid)
        self.nextButton.setEnabled(valid)
        self.setIndex(0 if valid else None)


    def setData(self, data):
        self.setNumberOfElements(len(data))


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
                info = "None"
            else:
                info = str(self.index) + "/" + str(self.numberOfElements)
            self.infoLabel.setText(info)
            
            self.selected.emit(self.index)


class QInputInfoBox(QLabel):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.input_text = "<b>Input image:</b><br>\n"

    def showInfo(self, name, shape):
        self.input_text = "<b>Input image:</b><br>\n"
        self.input_text += "Name: {}<br>\n".format(name)
        self.input_text += "Input shape: {}<br>\n".format(shape)
        self.setText(self.input_text)

