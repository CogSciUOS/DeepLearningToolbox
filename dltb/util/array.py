"""Utility functions to work with data arrays

There seem to be (at least) two formats in use in different
frameworks:

DATA_FORMAT_CHANNELS_FIRST:
    In this format, channel axis is stored before additional (spatial)
    axes, that is for 2-dimensional layers the axes would be ``(batch,
    channel, height, width)``.  Tensorflow used the string `NCHW` to
    indicate this data format.  In keras the value `channels_first` is
    returned by image_data_format()` to indicate this data format.

DATA_FORMAT_CHANNELS_LAST:
    This format stores the channel in the last axis of the data.
    That is, the axes for a 2-dimensional layer wourd be
    ``(batch, channel, height, width)``. Tensorflow used the
    string `NHWC` to indicate this data format, while
    Keras uses `channels_last`.

"""

# standard imports
from typing import Tuple, Union

# third party imports
import numpy as np


DATA_FORMAT_CHANNELS_FIRST = 'NCHW'
DATA_FORMAT_CHANNELS_LAST = 'NHWC'


def adapt_data_format(array_or_shape: Union[np.ndarray, Tuple[int]],
                      input_format: str = None, output_format: str = None,
                      add_batch: bool = False, remove_batch: bool = False,
                      batch: bool = None) -> Union[np.ndarray, Tuple[int]]:
    """Convert channel first to channel last format or vice versa.


    Parameters
    ----------
    array_or_shape
        The array or shape tuple to be converted.

    input_format
        Either 'channels_first' or 'channels_last'. If not given, opposite
        of `output_format` is used.

    output_format
        Either 'channels_first' or 'channels_last'. If not given, opposite
        of `input_format` is used.

    add_batch: bool
        Add a batch dimension to the data. This will usually be
        the first axis.

    remove_batch: bool
        Remove the batch dimension from the data.

    batch: bool
        A flag indicating if the data has a batch dimension. If
        no value is given, it is assumed that a batch dimension
        is present.

    Returns
    -------
    The converted numpy array.

    """
    if add_batch and remove_batch:
        raise ValueError("Simultaneous use of 'add_batch' and 'remove_batch'")

    is_array = isinstance(array_or_shape, np.ndarray)
    is_shape = isinstance(array_or_shape, tuple)

    if not is_shape and not is_array:
        raise TypeError("Input must be either tuple or ndarray "
                        f"but is {type(array_or_shape)}")

    # Check inputs.
    if add_batch:
        array_or_shape = (array_or_shape[np.newaxis] if is_array else
                          (None,) + array_or_shape)

    axes = len(array_or_shape) if is_shape else array_or_shape.ndim
    # if input_format in ('NCHW', 'NHWC') and axes != 4:
    #     raise ValueError(f"Data with {axes} axes "
    #                     f"{getattr(array_or_shape, 'shape', array_or_shape)}"
    #                     f" does not match '{input_format}'.")

    channel_axis = 0 if batch is False else 1
    if input_format in ('channels_first', 'NCHW'):
        input_format = DATA_FORMAT_CHANNELS_FIRST
    elif input_format in ('channels_last', 'NHWC'):
        input_format = DATA_FORMAT_CHANNELS_LAST
    if output_format in ('channels_first', 'NCHW'):
        output_format = DATA_FORMAT_CHANNELS_FIRST
    elif output_format in ('channels_last', 'NHWC'):
        output_format = DATA_FORMAT_CHANNELS_LAST

    # Do dataformat conversion based on the arguments.
    if input_format is output_format:
        pass  # No conversion needed to same input and output format.
    elif axes < 3:
        pass  # No needed of exchanging channels
    elif input_format is None or output_format is None:
        pass  # No conversion information is given
    elif (input_format is DATA_FORMAT_CHANNELS_FIRST and
          output_format is DATA_FORMAT_CHANNELS_LAST):
        if isinstance(array_or_shape, np.ndarray):
            array_or_shape = np.moveaxis(array_or_shape, channel_axis, -1)
        else: # isinstance(array_or_shape, np.ndarray):
            array_or_shape = (array_or_shape[0:channel_axis] +
                              array_or_shape[channel_axis+1:] +
                              array_or_shape[channel_axis:channel_axis+1])
    elif (input_format is DATA_FORMAT_CHANNELS_LAST and
          output_format is DATA_FORMAT_CHANNELS_FIRST):
        if isinstance(array_or_shape, np.ndarray):
            array_or_shape = np.moveaxis(array_or_shape, -1, channel_axis)
        else: # isinstance(array_or_shape, np.ndarray):
            array_or_shape = (array_or_shape[0:channel_axis] +
                              array_or_shape[-1:] +
                              array_or_shape[channel_axis:-1])
    else:
        raise ValueError(f"Cannot adapt data of shape {array_or_shape.shape} "
                         f"from '{input_format}' to '{output_format}'")

    # Removing batch dimension if desired
    if remove_batch:
        if isinstance(array_or_shape, np.ndarray):
            if len(array_or_shape.shape) != 1:
                raise ValueError("Cannot remove batch for a non-singleton"
                                 f"batch (shape={array_or_shape.shape})")
            array_or_shape = array_or_shape[0]
        else: # isinstance(array_or_shape, tuple):
            array_or_shape = array_or_shape[1:]

    return array_or_shape
