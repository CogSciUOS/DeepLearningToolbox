# FIXME[concept]: These is just a collection of first ideas for
# an abstract GUI class. Currently it is not used at all.
#
# Currently, the following methods are used by the toolbox:
#  
#
# The following methods may be called from the shell
#
#   gui.panel(name, create=True, show=True)
#
# Furthermore, the toolbox may provide methods to start/stop the GUI
#
#
# FIXME[old]: some other places refer to the GUI that should not do so:
# [grep -r 'gui\.'  | grep py: | grep -v qtgui/]
#  - base/observer.py:            if isinstance(self, locate('qtgui.utils.QObserver')):


class GUI:

    GUIs = {
        'tk': {
            'class': '',
            'requirements': ['tkinter']
        },
        'qt': {
            'class': '',
            'requirements': ['pyqt']
        },
        'gtk': {
            'class': '',
            'requirements': ['gtk']
        }
    }

    @staticmethod
    def create(id: str):
        if id not in GUI.GUIs:
            raise LookupException("No GUI called '{id}'.")
            

    def __init__(self):
        pass
