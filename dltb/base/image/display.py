"""Code for displaying images.
"""

# standard imports
from typing import Optional
from abc import ABC
import threading

# toolbox imports
from ..implementation import Implementable
from ..gui import SimpleDisplay, View
from ._base import LOG
from .image import Image, Imagelike, ImageGenerator


class ImageView(View):
    """A graphical element that can be used to view an image.
    This is an abstract class that should be inherited by
    some graphical user interface for example to provide
    a widget that can display images.

    Properties
    ----------
    image:
        An image currently viewed by this `ImageView`.  May be `None`.

    Arguments
    ---------
    image:
        An image to be assigned to this ImageView.
    """
    _image: Optional[Image] = None

    def __init__(self, image: Optional[Imagelike] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.image = image

    def __bool__(self) -> None:
        """An `ImageView` is considered `True` if it is currently
        displaying an `Image`.
        """
        return self._image is not None

    @property
    def image(self) -> Optional[Image]:
        """The image currently displayed in this `ImageDisplay`.
        """
        return self._image

    @image.setter
    def image(self, image: Optional[Imagelike]) -> None:
        if image is not self._image:
            self._set_image(Image(image))

    def _set_image(self, image: Optional[Image]) -> None:
        """Set a new image for this `ImageView`. Subclasses should overwrite
        this method if they need to perform special actions when a new
        image is assigned.

        Arguments
        ---------
        image:
            A new image (guaranteed to be different from the current image).
            May be `None`.
        """
        self._image = image


class ImageDisplay(SimpleDisplay, Implementable, ImageGenerator.Observer, ABC):
    """An :py:class:`ImageDisplay` is a `Display` intended to display images.
    Typically, it will use some graphical user interface to open a window in
    which the image is displayed. It may also provide some additional controls
    to adapt display properties.

    The `ImageDisplay` is responsible for running the graphical user
    interface, including an event loop, showing and hiding the window,
    blocking or non-blocking function calls, etc. functionality that
    is inherited from `Display`.  The actual presentation of the `Image`,
    including options for zooming, rotation, etc is done by the `view`
    of the `SimpleDisplay` which is expected to be a compatible
    `ImageView`.

    Usage scenarios
    ---------------
    >>> from dltb.base.image import ImageDisplay
    >>> imagelike = 'examples/reservoir-dogs.jpg'
    >>> display = ImageDisplay()

    Example 1: show an image in a window and block until the window is
    closed:

    >>> display.show(imagelike)  # blocking=True (default)

    Example 2: show an image in a window without blocking (the event loop
    for the window will be run in a separate thread):

    >>> display.show(imagelike, blocking=False)

    Example 3: show an image in a window without blocking. No event loop
    is started for the window and it is the caller's responsibility to
    regularly call display.process_events() to keep the interface
    responsive.

    >>> display.show(imagelike, blocking=None)
    >>> while display.opened:
    >>>    display.show(imagelike)  # implicitly calls display.process_events()

    Example 4: show an image for five seconds duration.
    After 5 seconds the display is closed.

    >>> display.show(imagelike, timeout=5.0)

    Example 5: show multiple images, each for five seconds, but don't close
    the window in between:

    >>> for image in images:
    >>>     display.show(image, timeout=5.0, close=False)
    >>> display.close()

    Example 6: presenter:

    >>> def presenter(display, video):
    >>>     for frame in video:
    >>>         if display.closed:
    >>>             break
    >>>         display.show(frame)
    >>>
    >>> display = Display()
    >>> display.present(presenter, (video,))

    """

    View = ImageView

    def __init__(self, image: Optional[Imagelike] = None, **kwargs) -> None:
        # pylint: disable=unused-argument
        super().__init__(**kwargs)
        self.image = image

    @property
    def image(self) -> Image:
        """The image currently displayed in this `ImageDisplay`.
        """
        return None if self.view is None else self.view.image

    @image.setter
    def image(self, image: Optional[Imagelike]) -> None:
        if self.view is None:
            raise RuntimeError("Cannot set image for ImageDisplay without "
                               "an ImageView.")
        LOG.info("ImageDisplay: Viewing image (%s).", type(image))
        self.view.image = image

    #
    # Overwriting private interface methods
    #

    def _show(self, *args, **kwargs) -> None:
        """Display the given image.

        Arguments
        ---------
        image: Imagelike
            The image to display. This may be a single image or a
            batch of images.
        """
        LOG.debug("ImageDisplay: _show: args=%s, kwargs=%s",
                  len(args), len(kwargs))
        if len(args) == 1: # and isinstance(args[0], get_args(Imagelike)):
            self.image = args[0]
        elif len(args) == 0:
            self.image = kwargs.get('image', None)
        else:
            raise ValueError(f"Too many positional arguments ({len(args)}) "
                             "to show in an ImageDisplay.")

    #
    # ImageObserver
    #

    def image_changed(self, tool: ImageGenerator, change) -> None:
        """Implementation of the :py:class:`ImageObserver` interface.
        The display will be updated if the image has changed.
        """
        if change.image_changed:
            running_in_main_thread = \
                threading.current_thread() is threading.main_thread()

            if running_in_main_thread or self.active:
                # avoid show starting the event loop
                self.show(tool.image, blocking=False)

    # run: There are different modes of running:
    #  - start loop in a background thread (that thread should
    #    have some method to stop it when the display is closed)
    #  - iteratively call the tool and show the result. At each step
    #    monitor self.active flag and stop if self.active is False
    #
    # currently used by ./demos/dl-styletransfer.py ...
    def run(self, tool: ImageGenerator) -> None:
        """Start an image generation tool in the background and run the event
        loop in the main thread.  Observe the image generator to
        display updated images when available.

        """
        running_in_main_thread = \
            threading.current_thread() is threading.main_thread()

        if not running_in_main_thread and not self.active:
            raise RuntimeError("Cannot open ImageDisplay from "
                               "background thread.")

        self.observe(tool, interests=ImageGenerator.Change('image_changed'))

        if running_in_main_thread:
            LOG.info("ImageDisplay: starting thread")
            try:
                # Start the tool in a background rhread
                thread = threading.Thread(target=tool.loop)
                thread.start()

                # Run the main event loop of the GUI to get a
                # responsive interface. This call will block until
                # the Display is stopped/closed.
                self.show(None, blocking=True)
                LOG.info("ImageDisplay: Application main event loop finished")
            except KeyboardInterrupt:
                LOG.warning("ImageDisplay: Keyboard interrupt.")
            finally:
                tool.stop()
                thread.join()
                LOG.info("ImageDisplay: thread joined")
        else:  # not running_in_main_thread
            # we are called from a background thread and hence can
            # simply run the tool.
            # FIXME[todo]: there should be a way to stop the tool,
            # once the Display is stopped/closed
            tool.loop()

        self.unobserve(tool)
