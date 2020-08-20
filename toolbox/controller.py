from typing import Iterable
from base import View as BaseView
from .toolbox import Toolbox
from network import Network
from datasource import Datasource

class OldView(BaseView, view_type=Toolbox):

    def __init__(self, toolbox=None, **kwargs):
        super().__init__(observable=toolbox, **kwargs)


from .toolbox import Toolbox
from base import Controller as BaseController



class OldController(OldView, BaseController):

    def __init__(self, toolbox=None, **kwargs):
        super().__init__(toolbox=toolbox, **kwargs)


