'''
.. moduleauthor:: Rasmus Diederichsen

.. module:: util

This module collects miscellaneous utilities.
'''

class ArgumentError(ValueError):
    '''Invalid argument exception'''
    pass

def grayscaleNormalized(array):
    '''Convert a float array to 8bit grayscale

    Parameters
    ----------
    array   :   np.ndarray
                Array of 2/3 dimensions and numeric dtype. In case of 3 dimensions, the image set is
                normalized globally.

    Returns
    -------
    np.ndarray
        Array mapped to [0,255]

    '''
    import numpy as np

    # normalization (values should be between 0 and 1)
    min_value = array.min()
    max_value = array.max()
    div = max(max_value - min_value, 1)
    return (((array - min_value) / div) * 255).astype(np.uint8)


class Identifiable:
    _id : str = None
    _counter : int = 0

    def __init__(self, id = None):
        if id is None:
            self._ensure_id()
        else:
            self._id = id

    def _ensure_id(self):
        if self._id is None:
            Identifiable._counter += 1
            self._id = self.__class__.__name__ + str(Identifiable._counter)
        return self._id
    
    def get_id(self):
        return self._ensure_id()

    def __hash__(self):
        return hash(self._ensure_id())
        
    def __eq__(self, other):
        if isinstance(other, Identifiable):
            return self._ensure_id() == other._ensure_id()
        return False
