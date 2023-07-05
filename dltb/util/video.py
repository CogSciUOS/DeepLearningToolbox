"""Utility functions for working with videos.
"""

# standard imports
from typing import Union
import time

# toolbox imports
from ..base.image import ImageDisplay
from ..base.video import VideoReader, VideoWriter, VideoDisplay
from .image import get_display
from .time import pacemaker


def copy(reader: VideoReader, writer: VideoWriter,
         transform=None, fps: float = None, progress=None) -> None:
    """Copy a video reader to a video writer.
    """
    # Set parameters of results
    #with VideoFileWriter(fps=30.0, size=(512, 288)) as writer:
    try:
        last_index = index = 0
        last_time = start_time = time.time()
        with reader as frames, writer as output:
            if fps is not None:
                frames = pacemaker(frames, 1.0/fps)

            if progress is not None:
                frames = progress(frames)
            for index, frame in enumerate(reader):
                if transform is not None:
                    frame = transform(frame)
                output += frame

                new_time = time.time()
                if new_time - last_time > 1.0:
                    print("Running at "
                          f"{index-last_index/(new_time-last_time):.2f}"
                          " frames per seccond")
                    last_time, last_index = new_time, index
    except KeyboardInterrupt:
        print("Keyboard interrupt.")
    finally:
        duration = time.time() - start_time
        print(f"Copied {index} frames in {duration:.2f} seconds"
              f" (avarage {index/duration} fps)")


def show(reader: VideoReader, fps: float = None,
         display: Union[VideoDisplay, ImageDisplay] = None,
         module: str = None) -> None:
    """Show the given video.


    Examples
    --------

    >>> from dltb.base.video import RandomReader
    >>> show(RandomReader())

    >>> from dltb.base.video import Webcam
    >>> show(Webcam())

    """
    if display is None:
        display = get_display(module=module)
    if isinstance(display, ImageDisplay):
        display = VideoDisplay(display=display)
    if not isinstance(display, VideoDisplay):
        raise ValueError("Could not determine a valide display.")

    copy(reader, display, fps=fps)
