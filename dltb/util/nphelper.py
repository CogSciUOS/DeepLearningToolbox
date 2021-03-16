"""General numpy helper functions.
"""
# FIXME[todo]: rename to multimax (and argmultimax) and introduce
# multimin (and argmultimin)

# third-party imports
import numpy as np


def top(array: np.ndarray, top: int = 1, axis: int = None,
        sort: bool = False) -> np.ndarray:
    """Get the `top` values of an array.

    Arguments
    ---------
    array:
        A be one- or two-dimensional array. In the two-dimensional
        case the shape is (indices, channels).
    top:
        The number of top values to return.
    axis:
        The axis along which the top elements are taken. If `None`,
        the array is treated as if it had first been flattened to 1d.
    sort:
        A flag indicating if the the top values are unsorted
        (`False`) or sorted (`True`).

    Result
    ------
    values:
        The `top` values in this array.
    """
    top = min(top, array.size if axis is None else array.shape[axis])
    top_unsorted = np.partition(-array, top, axis=axis)
    # FIXME[hack]: special treatment in case of axis == 1 should be
    # avoided or should be made more general; not axis covers
    # axis is None and axis == 0
    top_unsorted = \
        top_unsorted[:top] if not axis else top_unsorted[:, :top]
    return np.sort(top_unsorted, axis=axis) if sort else top_unsorted


def argtop(array: np.ndarray, top: int = 1, axis: int = None,
           sort: bool = False) -> np.ndarray:
    """Get indices for the `top` values of an array.

    Arguments
    ---------
    array:
        A be one- or two-dimensional array. In the two-dimensional
        case the shape is (indices, channels).
    top:
        The number of indices (per channel).
    axis:
        The axis along which the top elements are taken. If `None`,
        the array is treated as if it had first been flattened to 1d.
    sort:
        A flag indicating if the the top values are unsorted
        (`False`) or sorted (`True`).

    Result
    ------
    indices:
        The indices of the `top` values.
    """
    # get indices for top elements along axis
    # Remark: here we could use np.argsort(-array)[:n]
    # but that may be slow for a larger arrays, as it does a full sort.
    # The numpy.partition provides a faster, though somewhat more
    # complicated method.
    size = array.size if axis is None else array.shape[axis]
    if size == 1:
        return np.zeros(array.shape, dtype=np.int)
    top = min(top, size)
    top_indices_unsorted = np.argpartition(-array, top, axis=axis)
    # FIXME[hack]: special treatment in case of axis == 1 should be
    # avoided or should be made more general; not axis covers
    # axis is None and axis == 0
    top_indices_unsorted = top_indices_unsorted[:top] \
        if not axis else top_indices_unsorted[:, :top]

    if not sort:
        return top_indices_unsorted

    # get correspondig (unsorted) top values:
    top_unsorted = \
        np.take_along_axis(array, top_indices_unsorted, axis=axis)

    # sort top values
    top_order = np.argsort(-top_unsorted, axis=axis)

    # and return the corresponding indices
    return np.take_along_axis(top_indices_unsorted, top_order, axis=axis)
