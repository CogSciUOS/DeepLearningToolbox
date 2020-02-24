'''
.. module:: resize

This module defines the :py:class:`ShapeAdaptor` and
:py:class:`ResizePolicy` classes which can be used to wrap a
:py:class:`datasources.DataSource` object so that the items it yields
contain a resized version of the original image.

.. moduleauthor:: Rasmus Diederichsen
'''
from datasources import DataSource, InputData
from network import Network
import numpy as np


class _ResizePolicyBase(object):
    '''Base class defining common properties of all resizing policies.

    Attributes
    ----------
    _new_shape  :   tuple
                    The shape to convert images to. Will be stripped of singular dimensions
    '''
    _new_shape: tuple = None

    def setShape(self, new_shape):
        '''Set the shape to match. If the last dimension is 1, it is removed.

        Parameters
        ----------
        new_shape   :   tuple or list
                        Shape to match.

        Raises
        ------
        ValueError
            If leading dimension is None (aka batch)
        '''
        if new_shape[0] is None:
            raise ValueError('Cannot work with None dimensions')
        if new_shape[-1] == 1:
            # remove channel dim
            new_shape = new_shape[:-1]
        self._new_shape = new_shape

    def resize(self, image):
        '''Resize an image according to this policy.

        Parameters
        ----------
        image   :   np.ndarray
                    Image to resize
        '''
        raise NotImplementedError('Abstract base class ResizePolicy cannot be used directly.')


class ResizePolicyBilinear(_ResizePolicyBase):
    '''Resize policy which bilinearly interpolates images to the target size.'''

    def resize(self, img):
        from util.image import imresize

        if self._new_shape is None:
            return img
        
        if self._new_shape[0:2] == img.shape[0:2]:
            return img
        
        #return resize(img, self._new_shape, preserve_range=True)
        return imresize(img, self._new_shape[0:2])


class ResizePolicyPad(_ResizePolicyBase):
    '''Resize policy which will pad the input image to the target size. This will not work if
    the target size is smaller than the source size in any dimension.'''

    def __init__(self, mode, **kwargs):
        '''
        Parameters
        ----------
        mode    :   str
                    Any of the values excepted by :py:func:`np.pad`
        kwargs  :   dict
                    Additional arguments to pass to :py:func:`np.pad`, such as the value to pad with
        '''
        self._pad_mode = mode
        self._pad_kwargs = kwargs

    def resize(self, img):
        if self._new_shape is None or self._new_shape == img.shape:
            return img
        h, w = img.shape[:2]
        new_h, new_w = self._new_shape[:2]

        if new_h < h or new_w < w:
            raise ValueError('Cannot pad to a smaller size. Use `ResizePolicy.Crop` instead.')

        # necessary padding to reach desired size
        pad_h = new_h - h
        pad_w = new_w - w
        
        # If padding is not even, we put one more padding pixel to the
        # bottom/right
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        return np.pad(img, (top, bottom, left, right), self._pad_mode, **self._pad_kwargs)


class ResizePolicyChannels(_ResizePolicyBase):
    """Resize policy which adapts the number of channels.


    Attributes
    ----------
    _channels: int
        The number of (color) cannels.
    """

    _channels = None

    def setShape(self, new_shape):
        if len(new_shape) == 3:
            self._channels = new_shape[2]
        else:
            self._channels = None

    def resize(self, img):
        if self._channels is None:
            if img.ndim == 3 and img.shape[2] == 1:
                img = np.squeeze(img, axis=2)
            elif img.ndim == 3:
                # FIXME[hack]: find better way to do RGB <-> grayscale
                # conversion
                # FIXME[question]: what are we trying to achieve here?
                # We have self._channels is None, so why do we want
                # to reduce the number of channels?
                img = np.mean(img, axis=2).astype(np.uint8)
            elif img.ndim != 2:
                raise ValueError('Incompatible shape.')
        elif img.ndim == 2:
            if self._channels == 1:
                img = img[..., np.newaxis]
            else:
                # Blow up to three dimensions by repeating the channel
                img = img[..., np.newaxis].repeat(self._channels, axis=2)
        elif self._channels > 1 and img.shape[2] == 1:
            img = img.repeat(3, axis=2)
        elif self._channels == 1 and img.shape[2] > 1:
            # FIXME[hack]: find better way to do RGB <-> grayscale
            # conversion
            img = np.mean(img, axis=2, keepdims=True).astype(np.uint8)
        elif self._channels != img.shape[2]:
            raise ValueError('Incompatible network input shape: network expects {} channels, but image has shape {}.'.format(self._channels,img.shape))
        return img

class ResizePolicy(object):

    @staticmethod
    def Bilinear():
        '''Create a bilinear interpolation policy.'''
        return ResizePolicyBilinear()

    @staticmethod
    def Pad():
        '''Create a pad-with-zero policy.'''
        return ResizePolicyPad('constant', constant_values=0)

    @staticmethod
    def Crop():
        raise NotImplementedError

    @staticmethod
    def Channels():
        return ResizePolicyChannels()


class ShapeAdaptor:
    '''Adaptive wrapper around a :py:class:`DataSource`'''

    def __init__(self, resize_policy, network: Network=None):
        '''
        Parameters
        ----------
        resize  :   ResizePolicy
                    Policy to use for resizing images
        network :   network.Network
                    Network to adapt to
        '''
        self._resize = resize_policy
        self.setNetwork(network)

    def setNetwork(self, network: Network):
        '''Change the network to adapt to.

        Parameters
        ----------
        network :   network.Network
        '''
        if network is not None:
            self._resize.setShape(network.get_input_shape(include_batch=False))

    def __call__(self, data: np.ndarray):
        return self._resize.resize(data)
