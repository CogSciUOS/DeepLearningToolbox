import numpy as np

from .panel import Panel
from PyQt5.QtWidgets import QWidget
from qtgui.widgets import QMatrixView, QConnectionView

# FIXME[todo]: add docstrings!

class ExperimentsPanel(Panel):
    '''This Panel is intended for temporary experiments with the
    graphical components of the deep visualization toolbox.

    '''

    _network = None
    _layer = None
    _data = None
    
    def __init__(self, parent = None):
        '''Initialization of the ExperimentsView.

        Parameters
        ---------
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self.initUI()


    def initUI(self):

        # Correlation matrix view
        correlations = np.random.rand(64,64) * 2 - 1
        self.matrix_view = QMatrixView(correlations, self)
        self.matrix_view.move(10,10)
        self.matrix_view.resize(500,500)

        # FIXME[todo]: should not be necessary!
        self.matrix_view.update()


        #self.matrix_widget = MatrixWidget(correlations, self)
        #self.matrix_widget.move(600,10)
        #self.matrix_widget.resize(500,500)

        #self.matrix_widget.repaint()

        self.connections_view = QConnectionView(self)
        self.connections_view.move(550,10)
        self.connections_view.resize(500,500)


        
    def setNetwork(self, network = None) -> None:
        self._network = network
        self.updateActivation()
        
    def setLayer(self, layer = None) -> None:
        self._layer = layer
        self.updateActivation()
        
    def setInputData(self, data = None):
        '''Provide one data point as input for the network.
        '''
        self._data = data
        self.updateActivation()


    def updateActivation(self):

        if self._network is None:
            activations = None
        elif self._layer is None:
            activations = None
        elif self._data is None:
            activations = None
        else:
            activations = self._network.get_activations([self._layer],
                                                        self._data)[0]

        # FIXME[todo]: we actually need activation of two subsequent layers
        self.connections_view.setActivation(activations,activations)

