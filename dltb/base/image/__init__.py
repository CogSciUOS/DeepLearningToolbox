"""Defintion of abstract classes for image handling.

The central data structure is :py:class:`Image`, a subclass of
:py:class:`Data`, specialized to work with images.  It provides,
for example, properties like size and channels.

Relation to other `image` modules in the Deep Learning ToolBox:

* :py:mod:`dltb.util.image`: This defines general functions for image I/O and
  basic image operations. That module should be standalone, not
  (directly) requiring other parts of the toolbox (besides util) or
  third party modules (besides numpy). However, implementation for
  the interfaces defined there are provided by third party modules,
  which are automagically loaded if needed.

* :py:mod:`dltb.tool.image`: Extension of the :py:class:`Tool` API to provide
  a class :py:class:`ImageTool` which can work on `Image` data objects.
  So that module obviously depends on :py:mod:``dltb.base.image` and
  it may make use of functionality provided by :py:mod:`dltb.util.image`.
"""

# toolbox imports
from ._base import Imagelike
from .image import Image, ImageProperties
from .image import ImageExtension
from .image import ImageOperator, ImageGenerator, ImageObservable
from .image import ImageReader, ImageWriter
from .types import Size, Sizelike, Format, Colorspace, Color
from .types2 import BoundingBox, Region, PointsBasedLocation, Landmarks
from .transform import ImageResizer, ImageWarper
from .display import ImageView, ImageDisplay

# FIXME[todo]: create an interface to work with different image/data formats
# (as started in dltb.thirdparty.pil)
# * add a way to specify the default format for reading images
#   - in dltb.util.image.imread(format='pil')
#   - for Imagesources
# * add on the fly conversion for Data objects, e.g.
#   data.pil should
#   - check if property pil already exists
#   - if not: invoke Image.as_pil(data)
#   - store the result as property data.pil
#   - return it
# * this method could be extended:
#   - just store filename and load on demand
#   - compute size on demand
#
