# Image Processing in the Deep Learning ToolBox


## The type `Image` and `Imagelike`

The type expression `dltb.base.image.Imagelike` can be used for all
kind of data that can be interpreted as an image. This may be numpy
arrays, `Data` objects, or if URLs pointing to image files. The class
`Image` provides functions (like `as_data`, or `as_array`) to convert
an `Imagelike` into an image of the desired type.


## The module `dltb.base.image`

A collection of abstract base classes defining different types of
image manipulation. The ideas is that these clases are implemented
using thirdparty modules, abstracting away from the specific library
to be used, making it easy to switch between libraries and to work
with those that are available at the system.

## The module `dltb.util.image`

The module `dltb.util.image` provides a functional API for image
processing. Its main design goal is ease of use.

While the interface is held simple and purely functional, under the
hood it uses the full infrastructure of `dltb.base.image`, allowing
to make use of differnt Python image processing libraries.

## The class `dltb.tool.image.ImageTool`

Abstract base classes for tools that operate on images. This abstract
classes provide frequently used functionality, like resizing and
different forms of normalization. They also add image specific methods
to the interface, usually suffixed with `_image`, like `detect_image`
or `recognize_image`, allowing to directly invoke them with an
arbitrary `Imagelike` argument.


## FIXME[old]: The module `util.image`

Defines metadata classes for images and some auxiliary functions:
* move metadata to `dltb.data.image`
* move auxiliary functions to `dltb.util.image`


# Image tools


## Displaying images

### ImageObservable

The `image_changed` notification will be send whenever an image has
changed.



# Image libraries for Python

There are several Python packages around that provide function for
image I/O and image manipulation (last updated April 2020):

* imageio:
  - several image I/O functions, including video and webcam access
  - Python 3.5+
  - https://imageio.github.io/
  - https://github.com/imageio/imageio
  - https://imageio.readthedocs.io/en/stable/
  - Version 2.8.0 - February 2020

* cv2 (OpenCV):
  - functionality for image I/O, including video and webcam access (via ffmpeg)
  - resizing, drawing, ...
  - https://pypi.org/project/opencv-python
  - Version 4.2.0.32 - Februar 2020

* scikit-image:
  - Image processing algorithms for SciPy, including IO, morphology, filtering, warping, color manipulation, object detection, etc.
  - https://scikit-image.org/
  - https://pypi.org/project/scikit-image/
  - Version 0.16.2 - October 2019

* mahotas
  - Computer Vision in Python. It includes many algorithms implemented
    in C++ for speed while operating in numpy arrays and with a very
    clean Python interface.
  - https://github.com/luispedro/mahotas
  - https://mahotas.readthedocs.io/en/latest/
  - Version 1.4.8 - October 2019

* imutils: (by Adrian Rosebrock from PyImageSearch)
  - A series of convenience functions to make basic image processing
    functions such as translation, rotation, resizing,
    skeletonization, displaying Matplotlib images, sorting contours,
    detecting edges, and much more easier with OpenCV
  - https://github.com/jrosebr1/imutils
  - Version 0.5.3 - August 2019

* Pillow/PIL
  - Pillow is the friendly PIL fork by Alex Clark and Contributors.
  - PIL is the Python Imaging Library by Fredrik Lundh and Contributors.
  - Pillow uses its own Image class: PIL.Image.Image
  - https://pypi.org/project/Pillow/
  - Version 7.1.0 - April 2020
