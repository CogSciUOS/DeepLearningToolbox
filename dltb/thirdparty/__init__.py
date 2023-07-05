""".. moduleauthor:: Ulf Krumnack

.. module:: dltb.thirdparty

This package provides utility functions for checking dealing with
third-party libraries, ranging from an interface to check availability
and version matching, over package and data installation to abstract
interfaces that allow to access common functionionality (like images,
sound, video) using different libraries.


This module should not actually load the thirdparty modules but
only announce their availability to the program.  This includes:
* registering implementations provided by the modules.
* adding configuration properties used by the modules.
* adding command line options (usually via the configuration properties)

"""
# FIXME[todo]: It would be useful to extend the registration mechanism
# in a way that registers requirements for individual implementations
# (currently this is done by explicitly conditioning the registration
# using if ... blocks). A more systematic way would allow:
# * to show what implementations would be available (given the requiremnts
#   would be fulfilled)
# * to show what requirements have to be fulfilled
# * provide functions for installing requirements
# Requirements should not be restricted to (thirdparty) packages, but
# could also include
# * (github) repository
# * datasets
# * system resources (hardware and driver)


# standard imports
import logging

# toolbox imports
from ..config import config
from ..base.implementation import Implementable
from ..base.package import Package
from ..util.importer import importable, import_module, add_postimport_depency

# logging
LOG = logging.getLogger(__name__)

# THIRDPARTY: name of the Deep Learning Toolbox third-party module
THIRDPARTY = __name__  # 'dltb.thirdparty'

# DLTB: name of Deep Learning Toolbox package
DLTB = 'dltb'

#
# Register implementations for specific modules
#

IMAGE_READER = DLTB + '.base.image.ImageReader'
IMAGE_WRITER = DLTB + '.base.image.ImageWriter'
IMAGE_DISPLAY = DLTB + '.base.image.ImageDisplay'
IMAGE_RESIZER = DLTB + '.base.image.ImageResizer'
IMAGE_WARPER = DLTB + '.base.image.ImageWarper'
IMAGE_DETECTOR = DLTB + '.tool.detector.ImageDetector'

VIDEO_READER = DLTB + '.base.video.VideoReader'
VIDEO_WRITER = DLTB + '.base.video.VideoWriter'
VIDEO_WEBCAM = DLTB + '.base.video.Webcam'

SOUND_READER = DLTB + '.base.sound.SoundReader'
SOUND_WRITER = DLTB + '.base.sound.SoundWriter'
SOUND_DISPLAY = DLTB + '.base.sound.SoundDisplay'
SOUND_PLAYER = DLTB + '.base.sound.SoundPlayer'
SOUND_RECORDER = DLTB + '.base.sound.SoundRecorder'


FACE_ARCFACE = DLTB + '.tool.face.recognize.ArcFace'
FACE_DETECTOR = DLTB + '.tool.face.detector.Detector'
FACE_MTCNN = DLTB + '.tool.face.mtcnn.Detector'

IMAGE_GAN =  DLTB + '.tool.generator.ImageGAN'

YOUTUBE_DOWNLOADER =  DLTB + '.util.download.YoutubeDownloader'


#
# Registering Packages
#

Package(module='numpy', label='NumPy',
        description="NumPy is the fundamental package for "
        "scientific computing with Python.")
Package(module='tensorflow', label='TensorFlow',
        description="TensorFlow")
Package(module="keras", label='Keras')
Package(module="torch", label='Torch',
        description="The Torch deep learning library")
Package(module='appsdir', label='Appsdir')
Package(module='matplotlib', label='Matplotlib')
Package(module='cv2', label='OpenCV')
Package(module='caffe', label='Caffe')
Package(key='qt', module='PyQt5', label='Qt',
        description="Qt Widgets for creating graphical user interfaces.")
Package(module='pycuda', label='PyCuda',
        conda='pycuda', conda_channel='lukepfister')
Package(module='lucid', label='Lucid',
        description="A collection of infrastructure and tools "
        "for research in neural network interpretability.")
Package(module='imutils', label='Imutils',
        description='A series of convenience functions to make '
        'basic image processing functions such as translation, '
        'rotation, resizing, skeletonization, displaying '
        'Matplotlib images, sorting contours, detecting edges, '
        'and much more easier with OpenCV and both '
        'Python 2.7 and Python 3.')
Package(module='dlib', label='Dlib',
        description='Dlib is a modern C++ toolkit containing '
        'machine learning algorithms and tools '
        'for creating complex software to solve real world problems.')
Package(module='ikkuna', label='Ikkuna',
        description='A tool for monitoring neural network training.')
Package(module='sklearn', label='scikit-learn',
        description='Machine Learning in Python.')


if importable('tensorflow'):
    import_module('.tensorflow_register', __name__)

if importable('keras') or importable('tensorflow'):
    import_module('.keras_register', __name__)

if importable('torch'):
    import_module('.torch_register', __name__)

if importable('PIL'):
    add_postimport_depency('PIL', ('.pil', __name__))

if importable('sklearn'):
    add_postimport_depency('sklearn', ('.sklearn', __name__))

if importable('skimage'):
    SKIMAGE = THIRDPARTY + '.skimage'
    Implementable.register_module_alias(SKIMAGE, 'skimage')
    Implementable.register_module_alias(SKIMAGE, 'scikit-image')
    Implementable.register_implementation(IMAGE_RESIZER,
                                          SKIMAGE + '.ImageUtils')
    Implementable.register_implementation(IMAGE_WARPER,
                                          SKIMAGE + '.ImageUtils')

if importable('imageio'):  # numpy
    IMAGEIO = THIRDPARTY + '.imageio'
    Implementable.register_module_alias(THIRDPARTY + '.imageio', 'imageio')
    Implementable.register_implementation(IMAGE_READER,
                                          IMAGEIO + '.ImageIO')
    Implementable.register_implementation(IMAGE_WRITER,
                                          IMAGEIO + '.ImageIO')
    # the following require imageio_ffmpeg
    Implementable.register_implementation(VIDEO_READER,
                                          IMAGEIO + '.VideoReader')
    Implementable.register_implementation(VIDEO_WRITER,
                                          IMAGEIO + '.VideoWriter')
    Implementable.register_implementation(VIDEO_WEBCAM,
                                          IMAGEIO + '.Webcam')

if importable('matplotlib'):
    MATPLOTLIB = THIRDPARTY + '.matplotlib'
    Implementable.register_module_alias(MATPLOTLIB, 'plt')
    Implementable.register_module_alias(MATPLOTLIB, 'matplotlib')
    Implementable.register_implementation(IMAGE_READER,
                                          MATPLOTLIB + '.ImageIO')
    Implementable.register_implementation(IMAGE_WRITER,
                                          MATPLOTLIB + '.ImageIO')
    Implementable.register_implementation(IMAGE_DISPLAY, MATPLOTLIB +
                                          '.image.MplImageDisplay')
    Implementable.register_implementation(SOUND_DISPLAY, MATPLOTLIB +
                                          '.sound.MplSoundDisplay')
    Implementable.register_implementation(DLTB + '.util.plot.Display',
                                          MATPLOTLIB +'.MplDisplay')
    Implementable.register_implementation(DLTB +
                                          '.util.plot.TilingPlotter',
                                          MATPLOTLIB +'.MplTilingPlotter')
    Implementable.register_implementation(DLTB +
                                          '.util.plot.Scatter2dPlotter',
                                          MATPLOTLIB +'.MplScatter2dPlotter')

if importable('cv2'):
    OPENCV = THIRDPARTY + '.opencv'
    Implementable.register_module_alias(OPENCV, 'cv2')
    Implementable.register_module_alias(OPENCV, 'opencv')
    Implementable.register_implementation(IMAGE_READER,
                                          OPENCV + '.ImageIO')
    Implementable.register_implementation(IMAGE_WRITER,
                                          OPENCV + '.ImageIO')
    Implementable.register_implementation(IMAGE_DISPLAY,
                                          OPENCV + '.ImageDisplay')
    Implementable.register_implementation(IMAGE_RESIZER,
                                          OPENCV + '.ImageUtils')
    Implementable.register_implementation(IMAGE_WARPER,
                                          OPENCV + '.ImageUtils')
    Implementable.register_implementation(VIDEO_READER,
                                          OPENCV + '.VideoReader')
    Implementable.register_implementation(VIDEO_WRITER,
                                          OPENCV + '.VideoWriter')
    Implementable.register_implementation(VIDEO_WEBCAM,
                                          OPENCV + '.Webcam')
    Implementable.register_implementation(FACE_DETECTOR,
                                          OPENCV + '.face.DetectorHaar')
    Implementable.register_implementation(FACE_DETECTOR,
                                          OPENCV + '.face.DetectorSSD')

if importable('soundfile'):
    SOUNDFILE = THIRDPARTY + '.soundfile'
    Implementable.register_implementation(SOUND_READER,
                                          SOUNDFILE + '.SoundReader')
    Implementable.register_implementation(SOUND_WRITER,
                                          SOUNDFILE + '.SoundWriter')

if importable('librosa'):
    LIBROSA = THIRDPARTY + '.librosa'
    Implementable.register_implementation(SOUND_READER,
                                          LIBROSA + '.SoundReader')


if importable('sounddevice'):
    SOUNDDEVICE = THIRDPARTY + '.sounddevice'
    Implementable.register_implementation(SOUND_PLAYER,
                                          SOUNDDEVICE + '.SoundPlayer')
    Implementable.register_implementation(SOUND_RECORDER,
                                          SOUNDDEVICE + '.SoundRecorder')

if importable('PyQt5'):
    add_postimport_depency('PyQt5', ('.qt', THIRDPARTY))
    Implementable.register_module_alias(THIRDPARTY + '.qt', 'qt')
    Implementable.register_implementation(IMAGE_DISPLAY,
                                          THIRDPARTY + '.qt.ImageDisplay')

if importable('dlib') and importable('imutils'):
    DLIB = THIRDPARTY + '.dlib'
    Implementable.register_implementation(FACE_DETECTOR,
                                          DLIB + '.DetectorCNN')
    Implementable.register_implementation(FACE_DETECTOR,
                                          DLIB + '.DetectorHOG')

if importable('tensorflow'):
    Implementable.register_module_alias(THIRDPARTY + '.arcface', 'arcface')
    Implementable.register_implementation(FACE_ARCFACE,
                                          THIRDPARTY + '.arcface.ArcFace')

if importable('mtcnn') and importable('tensorflow'):
    Implementable.register_module_alias(THIRDPARTY + '.mtcnn', 'mtcnn')
    Implementable.register_implementation(FACE_MTCNN,
                                          THIRDPARTY + '.mtcnn.DetectorMTCNN')

if importable('torch'):  # requires the face_evolve repository
    Implementable.register_module_alias(THIRDPARTY + '.face_evolve.mtcnn',
                                        'mtcnn2')
    Implementable.register_implementation(FACE_MTCNN,
                                          THIRDPARTY +
                                          '.face_evolve.mtcnn.Detector')

if importable('tensorflow'):  # and 'dnnlib'
    Implementable.register_implementation(IMAGE_GAN,
                                          THIRDPARTY + '.nvlabs.StyleGAN')
    Implementable.register_implementation(IMAGE_GAN,
                                          THIRDPARTY + '.nvlabs.StyleGAN2')

if importable('nnable'):  # and 'nnabla_ext.cuda'
    Implementable.register_implementation(IMAGE_GAN,
                                          THIRDPARTY + '.nnabla.StyleGAN2')

if importable('torch'):
    Implementable.register_implementation(IMAGE_GAN,
                                          THIRDPARTY + '.experiments.VGAN')

if importable('pytube'):
    Implementable.register_implementation(YOUTUBE_DOWNLOADER,
                                          THIRDPARTY + '.pytube.PyTube')

if importable('mmvc') and importable('mmdet'):
    MMVC = THIRDPARTY + 'mmvc'
    Implementable.register_implementation(IMAGE_DETECTOR,
                                          MMVC + 'detector.Detector')
    print("Registered dltb.thirdparty.mmvc.detector.Detector")
    
# FIXME[todo]: there are also some stored models:
#  - models/example_caffe_network_deploy.prototxt
#  - models/example_keras_mnist_model.h5  <- network.examples.keras
#  - models/example_tf_alexnet/
#  - models/example_tf_mnist_model/
#  - models/example_torch_mnist_model.pth
#  - models/mnist.caffemodel

# FIXME[todo]: there are also examples in network/examples.py:
#  - keras  -> models/example_keras_mnist_model.h5
#  - torch

# -----------------------------------------------------------------------------
# Shield
if True:
    SHIELD = THIRDPARTY + '.shield'
    SHIELD_DETECTOR = SHIELD + '.ShieldDetector'
    Implementable.register_subclass(IMAGE_DETECTOR, SHIELD_DETECTOR)
    Implementable.register_implementation(SHIELD_DETECTOR,
                                          SHIELD + '.vehicle.Detector')
    print("Registered dltb.thirdparty.shield.vehicle.Detector")
    Implementable.register_implementation(SHIELD_DETECTOR,
                                          SHIELD + '.license_plate.Detector')
    print("Registered dltb.thirdparty.shield.license_plate.Detector")

# -----------------------------------------------------------------------------



#
# Check for some optional packages
#

# FIXME[old]: the following is quite old and too specific - it should
# be replaced by a more general requirements mechanism (see comment at
# the top of this file).

def warn_missing_dependencies():
    """Emit warnings concerning missing dependencies, i.e. third-party
    modules not install on this system.
    """
    if not importable('appdirs'):
        LOG.warning(
            "--------------------------------------------------------------\n"
            "info: module 'appdirs' is not installed.\n"
            "We can live without it, but having it around will provide\n"
            "additional features.\n"
            "See: https://github.com/ActiveState/appdirs\n"
            "--------------------------------------------------------------\n")

    if not importable('setproctitle'):
        LOG.warning(
            "--------------------------------------------------------------\n"
            "info: module 'setproctitle' is not installed.\n"
            "We can live without it, but having it around will provide\n"
            "additional features.\n"
            "See: https://github.com/dvarrazzo/py-setproctitle\n"
            "--------------------------------------------------------------\n")

if config.warn_missing_dependencies:
    warn_missing_dependencies()
