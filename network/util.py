from typing import Union

import numpy as np


def remove_batch_dimension(shape: tuple) -> tuple:
    """Set the batch dimension to None as it does not matter for the Network interface."""
    shape = list(shape)
    shape[0] = None
    shape = tuple(shape)
    return shape


def convert_data_format(array_or_shape: Union[np.ndarray, tuple],
                        input_format: str=None,
                        output_format: str=None) -> Union[np.ndarray, tuple]:
    """Convert channel first to channel last format or vice versa.

    Parameters
    ----------
    array_or_shape
        The array or shape tuple to be converted. Needs to be at least rank 3.
    input_format
        Either 'channels_first' or 'channels_last'. If not given, opposite of `output_format` is used.
    output_format
        Either 'channels_first' or 'channels_last'. If not given, opposite of `input_format` is used.

    Returns
    -------
    The converted numpy array.

    """
    is_tuple = False
    # Check inputs.
    if isinstance(array_or_shape, np.ndarray):
        if array_or_shape.ndim < 3:
            # raise ValueError('Tensor needs to be at least of rank 3 but is of rank {}.'.format(array_or_shape.ndim))
            # Non image arrays don't have to be converted.
            return array_or_shape
    elif isinstance(array_or_shape, tuple):
        # Convert to list for assignment later, but set a flag to remember it was a tuple.
        array_or_shape = list(array_or_shape)
        is_tuple = True
        if len(array_or_shape) < 3:
            # raise ValueError('Shape needs to be at least of rank 3 but is of rank {}.'.format(len(array_or_shape)))
            return array_or_shape
    else:
        raise TypeError('Input must be either tuple or ndarray but is {}'.format(type(array_or_shape)))
    # Do the conversion based on the arguments.
    if input_format == output_format:
        # No conversion needed to same input and output format.
        return array_or_shape
    elif ((input_format == 'channels_first' and output_format == 'channels_last') or
          (input_format == 'channels_first' and output_format is None) or
          (input_format is None and output_format == 'channels_last')):
        if isinstance(array_or_shape, np.ndarray):
            return np.moveaxis(array_or_shape, 1, -1)
        elif is_tuple:
            num_channels = array_or_shape[1]
            del array_or_shape[1]
            array_or_shape.append(num_channels)
            array_or_shape[-1] = num_channels
            return tuple(array_or_shape)
    elif ((input_format == 'channels_last' and output_format == 'channels_first') or
          (input_format == 'channels_last' and output_format is None) or
          (input_format is None and output_format == 'channels_first')):
        if isinstance(array_or_shape, np.ndarray):
            return np.moveaxis(array_or_shape, -1, 1)
        elif is_tuple:
            num_channels = array_or_shape[-1]
            del array_or_shape[-1]
            array_or_shape[1].insert(1, num_channels)
            return tuple(array_or_shape)
    else:
        raise ValueError('Data format must be either "channels_last" or "channels_first,"'
                         ' but is input_format="{}" and output_format="{}".'.format(input_format, output_format))

