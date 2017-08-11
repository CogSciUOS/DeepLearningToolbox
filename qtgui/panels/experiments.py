import numpy as np

from PyQt5.QtWidgets import QWidget
from qtgui.widgets import QMatrixView

# FIXME[todo]: add docstrings!

class ExperimentsPanel(QWidget):
    '''This Panel is intended for temporary experiments with the
    graphical components of the deep visualization toolbox.

    '''

    def __init__(self, parent = None):
        '''Initialization of the ExperimentsView.

        Arguments
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

