"""Integration of the PIL.Image (pillow) python image processing library.
Pillow provides its own image type (:py:class:`PIL.Image.Image`,
together with subclasses like :py:class:`PIL.ImageFile.ImageFile` and
:py:class:`PIL.JpegImagePlugin.JpegImageFile`). Some thirdparty libraries,
like `torchvision.transform` are based on this image type.

This module adds some hooks to work with pillow images:

* add an image format 'pil' that can be passed to
  :py:func:`dltb.util.image.imread` in order to obtain images in
  pillow format.

* adapt :py:func:`dltb.base.image.Image` to allow transformation of
  `Imagelike` to :py:class:`PIL.Image.Image` (as_pil) as well
  as and from :py:class:`PIL.Image.Image` to other formats.

"""
# FIXME[todo]:
# * add hook for dltb.util.image.imread(format='pil')
# * adapt the dltb.base.image.Imagelike datatype
# * add hook for dltb.base.image.Image.as_array()
# * add hook for dltb.base.image.Image.as_data()

# standard imports
import logging

# third party imports
import numpy as np
import PIL.Image

# toolbox imports
from datasource import Imagesource
from ..base.data import Data
from ..base.image import Image, Imagelike

# logging
LOG = logging.getLogger(__name__)


def as_pil(image: Imagelike, copy: bool = False) -> PIL.Image.Image:
    # pylint: disable=unused-argument
    """Get a :py:class:`PIL.Image.Image` from a given
    """
    if isinstance(image, PIL.Image.Image):
        return image

    if isinstance(image, str):
        return PIL.Image.open(image)

    if isinstance(image, Data):
        if not hasattr(image, 'pil'):
            image.add_attribute('pil', Image.as_pil(image.array))
        return image.pil

    if not isinstance(image, np.ndarray):
        image = Image.as_array(image)

    if issubclass(image.dtype.type, np.float):
        image = (image*255).astype('uint8')
    elif image.dtype != np.uint8:
        image = image.astype('uint8')
    return PIL.Image.fromarray(image, 'RGB')


LOG.info("Adapting dltb.base.image.Image: adding static method 'as_pil'")
Image.as_pil = staticmethod(as_pil)

Imagesource.add_loader('pil', PIL.Image.open)
