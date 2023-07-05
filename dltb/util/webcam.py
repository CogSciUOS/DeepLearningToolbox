"""Functional API to access webcam(s).
"""

# standard imports
import time

# toolbox imports
from ..base.image import ImageOperator 
# FIXME[bug]: we should be able to import generic webcam object
#from ..base.video import Webcam 
from ..base.video import VideoWriter 
from ..thirdparty.imageio import Webcam
from .image import imshow
from .time import pacemaker

def loop(device: int = None, operator: ImageOperator = None,
         writer: VideoWriter = None, fps: float = None):
    """Run a webcam loop. The loop will continously read frames from
    the webcam and process them. Processing may include filtering,
    displaying, and saving to a video.

    Arguments
    ---------
    webcam:
        Either a Webcam class or an instance of such a class.
    device:
        The device to use if a new Webcam object is created.
    operator:
        An image operator (filter) to be applied to the frames read
        from the webcam.
    display:
        An image display for showing the frames.
    writer:
        A :py:class:`VideoWriter` used for saving the frames to a
        video file.
    fps:
        Frames per second
    """
    # FIXME[todo]: arguments have to be implemented

    with Webcam(device=device) as webcam:  # FIXME[bug]: device should be optional
        frames = 0
        last_time = time.time()
        webcam_loop = webcam if fps is None else pacemaker(webcam, 1.0/fps)
        for frame in webcam_loop:
            frames += 1

            # apply image operator
            if operator is not None:
                frame = operator(frame)

            # display the image
            display = imshow(frame, blocking=None)

            if writer is not None:
                writer += frame

            new_time = time.time()
            if new_time - last_time > 1.0:
                print(f"Running at {frames/(new_time-last_time):.2f} "
                      "frames per seccond")
                last_time, frames = new_time, 0

            if display.closed:
                break


def snapshot(device: int = None):
    with Webcam(device=device) as webcam:
        frame = next(webcam)
    return frame
