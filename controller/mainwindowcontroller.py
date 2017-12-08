from network.network import Network

class MainWindowController(object):

    '''Controller class for the main GUI window. Will form the base handler for
    all events and aggregate subcontrollers for individual widgets.'''

    _current_panel = None
    _children   :   dict = {}
    # _model  :   model.Model = None

    def __init__(self):
        '''Create a new controller.'''
        pass

    def on_panel_selected(self, panel):
        self._panel = panel

    def add_child_controller(self, widget, controller):
        controller.setParent(self)
        self._children[widget] = controller

    def on_network_selected(self, network):
        self._model._network = network
        self._model.setChanged()

    def _saveState(self):
        pass

    def on_exit_clicked(self):
        from PyQt5.QtCore import QCoreApplication
        self._saveState()
        QCoreApplication.quit()

    def on_tab_selected(self, tab_id):
        pass
