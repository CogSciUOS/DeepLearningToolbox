

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
