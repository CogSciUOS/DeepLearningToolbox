'''
.. moduleauthor:: Rasmus Diederichsen

.. module:: util

This module collects miscellaneous utilities.
'''

class ArgumentError(ValueError):
    '''Invalid argument exception'''
    pass

def grayscale_normalized(array):
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
