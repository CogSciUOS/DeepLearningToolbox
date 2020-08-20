# standard imports
import os


class Directories:

    def __init__(self):
        self._map = {}
        home = os.environ['HOME']
        work = os.environ.get('WORK', home)
        space = os.environ.get('SPACE', home)
        self._map['temp'] = work
        self._map['models'] = os.path.join(space, 'models')
        self._map['data'] = os.path.join(space, 'data')
        self._map['opencv_models'] = \
            os.path.join(self._map['models'], 'opencv')
        self._map['tensorflow_models'] = \
            os.path.join(self._map['models'], 'tensorflow')

    def __getitem__(self, key: str) -> str:
        return self._map[key]
