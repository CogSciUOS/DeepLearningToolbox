import sys
import logging

try:
    from imageio import imread
except ImportError:
    try:
        from scipy.misc import imread
    except ImportError:
        try:
            from matplotlib.pyplot import imread
        except ImportError:
            # FIXME[hack]: better strategy to inform on missing modules
            explanation = ("Could not find any module providing 'imread'. "
                           "At least one such module is required "
                           "(e.g. imageio, scipy, matplotlib).")
            logging.fatal(explanation)
            sys.exit(1)
        # maybe also cv2, but convert the colorchannels

# FIXME[todo]: imresize:
#
# The documentation of scipy.misc.imresize says that imresize is
# deprecated! Use skimage.transform.resize instead. But it seems
# skimage.transform.resize gives different results from
# scipy.misc.imresize.
#  https://stackoverflow.com/questions/49374829/scipy-misc-imresize-deprecated-but-skimage-transform-resize-gives-different-resu
#
# -> Try using scipy.ndimage.interpolation.zoom()
#
# * cv2.resize(image,(width,height))
# * mahotas.imresize(img, nsize, order=3)
#   This function works in two ways: if nsize is a tuple or list of
#   integers, then the result will be of this size; otherwise, this
#   function behaves the same as mh.interpolate.zoom
#
# * mahotas.interpolate.zoom
# * imutils.resize

try:
    # rescale, resize, downscale_local_mean
    # image_rescaled = rescale(image, 0.25, anti_aliasing=False)
    # image_resized = resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)
    # image_downscaled = downscale_local_mean(image, (4, 3))
    from skimage.transform import resize as imresize
except ImportError:
    try:
        from scipy.misc import imresize
    except ImportError:
        # FIXME[hack]: better strategy to inform on missing modules
        explanation = ("Could not find any module providing 'imresize'. "
                       "At least one such module is required "
                       "(e.g. skimage, scipy, cv2).")
        logging.fatal(explanation)
        sys.exit(1)
