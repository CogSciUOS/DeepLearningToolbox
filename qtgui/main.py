import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QMainWindow, QTabWidget

from qtgui.panels import ActivationsPanel, ExperimentsPanel

# FIXME[todo]: add docstrings!

class DeepVisMainWindow(QMainWindow):
    '''The main window of the deep visualization toolbox.  The window is
    intended to hold different panels that allow for different kinds
    of visualizations.

    This class also provides central entry points to programmatically
    set certain aspects, like loading a network or input data,
    switching between different panels, etc.
    '''

    def __init__(self, network=None, data=None):
        super().__init__()

        # FIXME[matplotlib]: only needed if using matplotlib for ploting ...
        # prepare matplotlib for interactive plotting on the screen
        plt.ion()

        self.title = 'Activations'

        self.left = 10
        self.top = 10
        self.width = 1800
        self.height = 900

        self.initUI()


    def initUI(self):
        '''Initialize the graphical components of this user interface.
        '''

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.activations = ActivationsPanel()
        self.experiments = ExperimentsPanel()

        self.tabs = QTabWidget(self);
        # FIXME[layout]
        #self.tabs.setFixedSize(self.width-5, self.height-5)
        self.tabs.addTab(self.activations, "Main")
        self.tabs.addTab(self.experiments, "Experiments")

        # FIXME[question]: what is this?
        self.setCentralWidget(self.tabs)


    # FIXME[hack]: split up into sensible functions or rename ...
    def setNetwork(self, network, data):
        self.activations.addNetwork(network)
        self.activations.setInputData(data)
        self.update()
