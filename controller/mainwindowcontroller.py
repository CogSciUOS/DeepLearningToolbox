from network.network import Network
import qtgui

class MainWindowController(object):
    '''Controller class for the main GUI window. Will form the base handler for
    all events and aggregate subcontrollers for individual widgets.'''

    _current_panel : 'qtgui.panels.Panel' = None
    _children   :   dict = {}

    def __init__(self):
        '''Create a new controller.'''
        pass

    def on_panel_selected(self, panel : 'qtgui.panels.Panel'):
        '''Callback for selecting a new panel in the main window.

        Parameters
        ----------
        panel   :   qtgui.panels.Panel
                    The newly selected panel
        '''
        self._panel = panel

    def add_child_controller(self, widget, controller):
        '''Add a controller responsible for a subelement.

        Parameters
        ----------
        widget  :   PyQt5.QtWidgets.QWidget
                    Widget controlled by `controller`
        controller  :   controller.NetwworkController
                        Controller for the widgets
        '''
        controller.setParent(self)
        self._children[widget] = controller

    def on_network_selected(self, network):
        '''Callback for selecting a new network. (This may be unnecessary)

        Parameters
        ----------
        network :   network.Network
                    Newly selected network
        '''
        self._model.setNetwork(network)

    def _saveState(self):
        '''Callback for saving any application state inb4 quitting.'''
        pass

    def on_exit_clicked(self):
        '''Callback for clicking the exit button. This will save state and then terminate the Qt
        application'''
        from PyQt5.QtCore import QCoreApplication
        self._saveState()
        QCoreApplication.quit()
