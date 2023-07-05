"""General numpy helper functions.
"""
# standard imports
from typing import Optional

# third-party imports
import numpy as np

# since numpy 1.20 the use of np.float (and other numpy types) is deprecated
# in favor of using the Python builtin `float`.
# We will define some portability definitions, that allows to be compatible
# with old and new versions of numpy.
np_float = float  # pylint: disable=invalid-name
np_int = int  # pylint: disable=invalid-name
np_bool = bool  # pylint: disable=invalid-name
np_object = object  # pylint: disable=invalid-name
# This change in numpy also cause deprecation warning when using
# some (older) thirdparty modules:
#  - tensorboard 2.4.0: np.bool, np.object
#    (resolved in tensorboard 2.6.0)
#  - h5py 2.10.0: `np.typeDict` is a deprecated alias for `np.sctypeDict`.


def multimin(array: np.ndarray, num: int = 1, axis: Optional[int] = None,
             sort: bool = False) -> np.ndarray:
    """Get the `num` smallest values of an array.

    Arguments
    ---------
    array:
        A be one- or two-dimensional array. In the two-dimensional
        case the shape is (indices, channels).
    num:
        The number of smallest values to return.
    axis:
        The axis along which the smallest elements are taken. If `None`,
        the array is treated as if it had first been flattened to 1d.
    sort:
        A flag indicating if the values are unsorted
        (`False`) or sorted (`True`).

    Result
    ------
    values:
        The `num` smallest values in this array.
    """
    length = array.size if axis is None else array.shape[axis]
    num = min(num, length)
    min_unsorted = np.partition(array, num, axis=axis) \
        if num < length else array
    # FIXME[hack]: special treatment in case of axis == 1 should be
    # avoided or should be made more general; currently does not covers
    # axis is None and axis == 0!
    min_unsorted = \
        min_unsorted[:num] if not axis else min_unsorted[:, :num]
    return np.sort(min_unsorted, axis=axis) if sort else min_unsorted


def multimax(array: np.ndarray, num: int = 1, axis: Optional[int] = None,
             sort: bool = False) -> np.ndarray:
    """Get the `num` largest values of an array.

    Arguments
    ---------
    array:
        A be one- or two-dimensional array. In the two-dimensional
        case the shape is (indices, channels).
    num:
        The number of largest values to return.
    axis:
        The axis along which the smallest elements are taken. If `None`,
        the array is treated as if it had first been flattened to 1d.
    sort:
        A flag indicating if the maximal values are unsorted
        (`False`) or sorted (`True`).

    Result
    ------
    values:
        The `num` largest values in this array.
    """
    return -multimin(-array, num=num, axis=axis, sort=sort)


def argmultimin(array: np.ndarray, num: int = 1, axis: Optional[int] = None,
                sort: bool = False) -> np.ndarray:
    # get indices for num smallest elements along axis
    # Remark: here we could use np.argsort(-array)[:n]
    # but that may be slow for a larger arrays, as it does a full sort.
    # The numpy.partition provides a faster, though somewhat more
    # complicated method.
    size = array.size if axis is None else array.shape[axis]
    if size == 1:
        return np.zeros(array.shape, dtype=np.int)
    num = min(num, size)
    if num < size:  # np.arpartition requires num < array.shape[axis]
        min_indices_unsorted = np.argpartition(array, num, axis=axis)
    else:  # hack: artificially create full index array
        min_indices_unsorted = \
            np.meshgrid(*[np.arange(x) for x in array.shape])[axis].T
    # FIXME[hack]: special treatment in case of axis == 1 should be
    # avoided or should be made more general; not axis covers
    # axis is None and axis == 0
    min_indices_unsorted = min_indices_unsorted[:num] \
        if not axis else min_indices_unsorted[:, :num]

    if not sort:
        return min_indices_unsorted

    # get correspondig (unsorted) top values:
    min_unsorted = \
        np.take_along_axis(array, min_indices_unsorted, axis=axis)

    # sort top values
    top_order = np.argsort(min_unsorted, axis=axis)

    # and return the corresponding indices
    return np.take_along_axis(min_indices_unsorted, top_order, axis=axis)


def argmultimax(array: np.ndarray, num: int = 1, axis: Optional[int] = None,
                sort: bool = False) -> np.ndarray:
    """Get indices for the `num` largest values of an array.

    Arguments
    ---------
    array:
        A be one- or two-dimensional array. In the two-dimensional
        case the shape is (indices, channels).
    num:
        The number of indices (per channel).
    axis:
        The axis along which the top elements are taken. If `None`,
        the array is treated as if it had first been flattened to 1d.
    sort:
        A flag indicating if the the values are unsorted
        (`False`) or sorted (`True`).

    Result
    ------
    indices:
        The indices of the `top` values.
    """
    return argmultimin(-array, num=num, axis=axis, sort=sort)
