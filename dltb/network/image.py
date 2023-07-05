"""Networks to operate on image input.
"""

# standard imports
from typing import Tuple, Any, Union, List

# third party imports
import numpy as np

# toolbox imports
from .network import Network, LOG
from .base import Layerlike, as_layer
from ..base.data import Data
from ..base.image import Imagelike, ImageExtension
from ..tool.image import ImageTool
from ..util.image import imresize
from ..util.array import DATA_FORMAT_CHANNELS_LAST


class ImageNetwork(ImageExtension, ImageTool, base=Network):
    """A network for image processiong. Such a network provides
    additional methods to support passing images as arguments.

    Working on images introduces some additional concepts:

    size:
        An image typically has a specific size, given by width
        and height in pixels.
    """
    @property
    def input_size(self) -> Tuple[int, int]:
        """Size for input images.  May be `None` if the Network can
        operate on various input resolutions.
        """
        return self.get_input_shape()[1:-1]

    # FIXME[hack]: this changes the semantics of the function:
    # the base class expects inputs: np.ndarray, while we expect an Imagelike.
    # We should improve the interface (either a new method or a something
    # more flexible)
    def get_activations(self, inputs: Imagelike,
                        layer_ids: Any = None,
                        data_format: str = None,
                        as_dict: bool = False
                        ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get the activations for a given Imagelike Object.

        Arguments
        ---------
        inputs:
            The input image(s).
        layer_ids:
            Layers for which activations should be computed.
        data_format:
        as_dict:
        """

        LOG.debug("ImageNetwork.getActivations: inputs=%s [%s], layers=%s",
                  type(inputs), DATA_FORMAT_CHANNELS_LAST, layer_ids)
        internal = self.image_to_internal(inputs)
        is_list = isinstance(layer_ids, list)
        # batched: True = the input data where given a batch and hence
        # the result will be batch(es) of activation values
        # (the activation arrays have a batch dimension as their first axis)
        batched = (inputs.is_batch if isinstance(inputs, Data) else
                   isinstance(inputs, list))

        # Check whether the layer_ids are actually a list.
        layer_ids, is_list = self._force_list(layer_ids)

        # Transform the input_sample appropriate for the loaded_network.
        LOG.debug("ImageNetwork.getActivations: internal=%s (%s)",
                  internal.shape, self._internal_format)
        activations = self._get_activations(internal, layer_ids)
        LOG.debug("ImageNetwork.getActivations: internal activations=%s (%s)",
                  len(activations), self._internal_format)

        # Transform the output to stick to the canocial interface.
        activations = [self._transform_outputs(activation, data_format,
                                               unbatch=not batched,
                                               internal=False)
                       for activation in activations]

        LOG.debug("ImageNetwork.getActivations: output activations=%s (%s/%s)",
                  #activations[0].shape, data_format, self.data_format)
                  len(activations), data_format, self.data_format)

        # If it was just asked for the activations of a single layer,
        # return just an array.
        if not is_list:
            activations = activations[0]
        elif as_dict:
            activations = dict(zip(layer_ids, activations))
        return activations

    #
    # Implementation of the Tool interface (not used yet)
    #

    def extract_receptive_field(self, layer: Layerlike, unit: Tuple[int],
                                image: Imagelike) -> Imagelike:
        """Extract the receptive field for a unit in this :py:class:`Network`
        from an input image.

        Arguments
        ---------
        layer:
            The layer of the unit.
        unit:
            Coordinates for the unit in the `layer`. These may or may not
            include the channel (the channel does not influence the
            receptive field).
        image:
            The image from which the receptive field should be extracted.
            The image will undergo the same preprocessing (resizing/cropping),
            as it would undergo if the image would be processed by
            this :py:class:`Network`.

        Result
        ------
        extract:
            The part of the image in the receptive field of the unit,
            resized to fit the native input resolution of this
            :py:class:`Network`.
        """
        layer = as_layer(layer)
        resized = self.resize(image)
        (fr1, fc1), (fr2, fc2) = layer.receptive_field(unit)
        extract_shape = (fr2 - fr1, fc2 - fc1)
        if resized.ndim == 3:  # add color channel
            extract_shape += (resized.shape[-1], )
        extract = np.zeros(extract_shape)
        sr1, tr1 = max(fr1, 0), max(-fr1, 0)
        sc1, tc1 = max(fc1, 0), max(-fc1, 0)
        sr2, tr2 = min(fr2, resized.shape[0]), \
            extract_shape[0] + min(0, resized.shape[0] - fr2)
        sc2, tc2 = min(fc2, resized.shape[1]), \
            extract_shape[1] + min(0, resized.shape[1] - fc2)
        # print(f"field: [{fr1}:{fr2}, {fc1}:{fc2}] ({fr2-fr1}x{fc2-fc1}), "
        #       f"extract: {extract_shape[:2]}, "
        #       f"source:[{sr1}:{sr2}, {sc1}:{sc2}] ({sr2-sr1}x{sc2-sc1}), "
        #       f"target:[{tr1}:{tr2}, {tc1}:{tc2}] ({tr2-tr1}x{tc2-tc1})")
        extract[tr1:tr2, tc1:tc2] = resized[sr1:sr2, sc1:sc2]
        return extract

    def resize(self, image: Imagelike) -> Imagelike:
        """Resize the given image to match the (preferred) input
        size for this `ImageNetwork`.
        """
        # FIXME[hack]: this should be integrated into (or make use of)
        # the preprocessing logic
        return imresize(image, self.input_size)

    def image_to_internal(self, image: Imagelike) -> Any:
        """Optain the internal representation for the given
        image.
        """
        return self._image_to_internal(image)[np.newaxis]
        # to be implemented by subclasses

    def internal_to_image(self, data: Any) -> Imagelike:
        """Obtain an image from the internal representation.
        """
        return self._internal_to_image(data)
        # to be implemented by subclasses
