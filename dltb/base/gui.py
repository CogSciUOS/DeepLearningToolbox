"""Definition of base classes for a graphical user interface.


display = Display()
display.show(data)  # blocking=True

display = Display()

display.show(data)  # blocking=True


FIXME[todo]: not implemented yet

display = Display()
display.run(data_generator)


FIXME[todo]: allow to simply open display, show, run event loop and close

  Display(show=data)
  Display(run=data_generator)
  Display(run=callback, run_args=(...))

"""
# FIXME[design]: there should be a clearer API with respect to threads:
#  * run/stop the event loop
#    start:
#    - without calling show
#    stop:
#    - via gui
#    - via keyboard interrupt
#    - from a background thread
#  * decouple event loop functionality from show()
#    - simple mode: calling show (from main thread) may start
#      the event loop -> stop event loop upon close
#    - allow to 'show()' something from a background thread
#      (assuming a event loop is running)
#
#  * maybe move some of the arguments from show to __init__
#    e.g., allowing to start the event loop upon construction.

# standard imports
from typing import Optional, Union, List, Callable
from abc import ABC
import time
import logging
import threading

# third-party imports

# toolbox imports
from ..util.error import handle_exception

# logging
LOG = logging.getLogger(__name__)


class Display:
    """Base class for a display in a graphical user interface.
    A `Display` displays some content and typically runs an event loop
    to react to user interaction and update displayed elements.

    FIXME[concept]: The following points need to be worked out:
    * What to display? Images, Graphs, Plots, extra information.
      The content is not provided by the Display, but by some
      content provider, e.g., a :py:class:`View`. Plotter, ...
      There may be simple displays, just displaying one component,
      or multi-displays, displaying multiple components
    * How to display? A display can be seen as a standalone alternative
      to a complex graphical user interface.  It just manages
      the user interaction (event loop) but is not involved in the content.
      Convenience functions like `imshow`, `vidshow`, etc, may
      open their own display.
    * When to display? dynamic aspects of updating the display.
      Content provider should be able to inform the display when
      new content is available. Alternatively, the Display may
      periodically ask content providers if new content is
      available.

    Blocking and non-blocking display
    ---------------------------------

    There are two ways how an display can be used.  In blocking
    mode the execution of the main program is paused during display
    and is only continued when the display is closed.  In
    non-blocking mode, the the execution of the main program is
    continued while the display is shown.

    The blocking behaviour can be controlled by the `blocking`
    argument. It can be set to `True` (running the GUI event loop in
    the calling thread and thereby blocking it) or `False` (running
    the GUI event loop in some other thread).  It can also be set to
    `None` (meaning that no GUI event loop is started, which is
    similar to the non-blocking mode, however it will usually result
    in an inresponsive display window if no additional actions are
    undertaken; see the section on "GUI Event loop" below for more
    information).

    It is also possible to provide a function that will be run in a
    background thread (while the GUI event loop is run in the main
    thread, which usually is the better solution). The display will
    then run until the that background thread has ended.  The
    background thread can then call `display.show()` (without
    `blocking` argument) or some more specific functions to update the
    `Display`. It is the responsibility of that function to regularly
    check if the display is still open (has not been closed be the user
    or some other thread) by inspecting the :py:prop:`opened` property.
    So a typical (simplified) background function could look like that:

        def play(video, display):
            while display.opened:
                display.show(next(video))

    This method could than be registered by passing
    `display.show(blocking=lambda: play(video, display, ...)`.


    Ending the display
    ------------------

    Different conditions can be set up to determine when the display
    should end.  The most natural one is to wait until the display
    window is closed (using the standard controls of the window
    system). Additionally, the display can be terminated when a key is
    pressed on the keyboard or after a given amount of time.
    If run in a multi-threaded setting, it is also possible to end
    the `Display` programatically, calling :py:meth:`close`.

    The next question is: what should happen once the display ended?
    Again the most natural way is to close the window.  However,
    if more content is going to be displayed it may be more suitable
    to leave the window on screen and just remove the old content,
    until the next content is available.

    GUI Event loop
    --------------

    A :py:class:`Display` utilizes some graphical user interface
    (GUI).  Such a GUI usually requires to run an event loop to stay
    responsive, that is to react to mouse and other actions, like
    resizing, closing and even repainting the window. The event loop
    regularly checks if such events have occured and processes
    them.  Running a display without an event loop usually results in
    unpleasant behaviour and hence should be avoided.

    Nevertheless, running an event loop is not always straight
    forward.  Different GUI libraries use different concepts. For
    example, some libraries require that event loops are run in the
    main thread of the application, which can not always be realized
    (for example, it would not be possible to realize a non-blocking
    display in the main thread).  The :py:class:`Display` provides
    different means to deal with such problems.

    """

    _event_loop: Optional[threading.Thread]

    def __init__(self, module: Union[str, List[str]] = None, **kwargs) -> None:
        # pylint: disable=unused-argument
        super().__init__(**kwargs)

        # _opened: a flag indicating the current state of the Display.
        # True means that the display is ready to be used.
        # An open Display may or may not be visible on screen, it is
        # only ensured to be visible once it is active.
        self._opened: bool = False

        # _active: an event loop is running managing this Display
        self._active: bool = False

        # _blocking: a flag indicating if the display is currently operating
        # in blocking mode (True) or non-blocking mode (False).
        self._blocking: bool = None

        # _entered: a counter to for tracing how often the context manager
        # is used (usually it should only be used once!)
        self._entered: int = 0

        # _event_loop: some Thread object, referring to the thread running the
        # event loop.  If None, then currently no event loop is running.
        self._event_loop = None

        # _presentation: a Thread object running a presentation, initiated
        # by the method `present`
        self._presentation: Optional[threading.Thread] = None

        # _on_close: a hook to be called on close. May be an exception
        # which will be raised when the window is closed.
        self._on_close = None

    #
    # context manager
    #

    def __enter__(self) -> 'Display':
        self._entered += 1
        if self._entered > 1:
            LOG.warning("Entering Display multiple times: %d", self._entered)
        else:
            LOG.debug("Entering Display")
        self.open()
        return self

    def __exit__(self, exception_type, exception_value, _traceback) -> bool:
        LOG.debug("Exiting Display (%d, exceptions=%s)", self._entered,
                  exception_type)
        self._entered -= 1
        if self._entered == 0:
            self.close()
        if exception_value is not None:
            ok = exception_value is self._on_close
            self._on_close = None
            return ok
        return False

    #
    # public interface
    #

    def show(self, *args, blocking: Optional[bool] = True,
             close: Optional[bool] = None,
             timeout: Optional[float] = None,
             on_close: Optional[Union[Callable, type]] = None,
             **kwargs) -> None:
        """Show (or update) this Display.

        This method may start an event loop, depending on its current
        state and the value of the `blocking` argument.  This argument
        also determins if the method pauses execution of the main
        program during display or returns immediately.

        Arguments
        ---------
        blocking: bool
            A flag indicating if the display should be shown in blocking
            mode (`True`) or non-blocking mode (`False`).  If a `Callable`
            is provided, it will be run in a new Thread and can regularly
            update the display. The `Callable` should regularly check the
            :py:prop:`active` method and stop its operation once it
            is `False`. If `blocking` is `None`, no event loop is started,
            and it is the responsibility of the caller to regularly call
            :py:meth:`process_events` to keep the interface responsive.
        close: bool
            A flag indicating if the display should be closed after
            showing. Closing the display will also end all event
            loops that are running. If no value is provided, the
            display will be kept open, if it was already open when
            this method is called, and it will be closed in case it
            was closed before.
        wait_for_key: bool
            A flag indicating if the display should pause execution
            and wait or a key press.
        timeout: float
            Time in seconds to pause execution.

        """
        # If already active (event loop is running), the value of
        # blocking is irrelevant - we will keep the event loop running
        # and just update the display
        if self.opened or self.active or self._blocking is not None:
            blocking = None
        else:
            blocking = True if blocking is None else blocking

        # Should we close the Display at the end of this function?
        # We will do so, if the Display was closed at time of calling,
        # and we are not returning immediately (i.e., we are running in
        # blocking mode).
        if close is None:
            close = self.closed and (blocking is True)

        if on_close is not None:
            self._on_close = on_close

        # make sure the window is open
        if self.closed:
            # A background presentation may ignore that the Display
            # was closed and is still trying to update it by calling the
            # show method. We will stop such rude behaviour by raising an
            # exception:
            if self._presentation is threading.current_thread():
                raise RuntimeError("Presentation is trying to use a closed "
                                   "Display.")

            # In all other cases, we will open the display. Opening the
            # Display should allow to perform display operations. It may
            # already show the (empty) Display on screen and it may
            # already start an event loop, but neither of this is required.
            # It must, however, not block execution.
            self.open()

        # perform the actual show. Call the display specific
        # _show method to update the display with the remaining arguments.
        LOG.debug("Showing display: args=%s, kwargs=%s",
                  len(args), len(kwargs))
        self._show(*args, **kwargs)

        # run the event loop: now that the display is set up, we start
        # the event loop get a responsive GUI.
        LOG.debug("Display event handling: blocking=%s, opened=%s, active=%s, "
                  "event loop=%s, presentation=%s, timout=%s, closed=%s, "
                  "close=%s",
                  blocking, self.opened, self.active,
                  self._event_loop_is_running(),
                  self._presentation is not None,
                  timeout, self.closed, close)
        if blocking is True or isinstance(blocking, Callable):
            if isinstance(blocking, Callable):
                LOG.info("Display: starting presenter in background thread")
                self.present(blocking)
            if not self._event_loop_is_running() is None:
                self._active = True
                # now we start the event loop. This will block execution,
                # untilt that loop eventually stops.
                LOG.info("Display: starting event loop (blocking)")
                self._run_blocking_event_loop(timeout=timeout)
        elif blocking is False:
            if timeout is not None:
                LOG.warning("Setting timeout (%f) has no effect "
                            " for non-blocking image Display", timeout)
            if not self._event_loop_is_running():
                LOG.info("Display: starting event loop (non-blocking)")
                self._active = True
                self._run_nonblocking_event_loop()
            else:
                self._process_events()  # FIXME[hack]: should not be necessary
        elif blocking is None:  #  and not self.active:
            self._process_events()
        LOG.debug("Display: after event handling: blocking=%s, opened=%s, "
                  "active=%s, event loop=%s, presentation=%s, timout=%s, "
                  "closed=%s, close=%s", blocking, self.opened, self.active,
                  self._event_loop_is_running(),
                  self._presentation is not None,
                  timeout, self.closed, close)

        # close the window if desired
        if close:
            if self._entered > 0:
                LOG.warning("Closing image Display inside a context manager.")
            self.close()

    def open(self) -> None:
        """Open this :py:class:`Display`, meaning it will ensure that the
        display is visible on screen.  After successully opening the
        `Display`, the :py:prop:`opened` method should be `True`.

        Opening the `Display` will not automatically start an event
        loop. For opening the `Display` and starting an event loop,
        use the :py:class:`show` method.
        """
        if not self._opened and self._presentation is None:
            self._open()
            self._opened = True

    def close(self) -> None:
        """Close this :py:class:`Display`, meaning to make it disappear from
        screen (in case it is using its own window).  Closing the
        `Display` will also stop any event loop.
        """
        LOG.info("Closing Display "
                 "(opened=%s, presentation=%s, event loop=%s)",
                 self._opened, self._presentation is not None,
                 self._event_loop_is_running())
        if self._opened:
            self._active = False
            self._opened = False
            self._close()

        if self._presentation is not threading.current_thread():
            # we have started a presentation in a background Thread and
            # hence we will wait that this presentation finishes. In
            # order for this to work smoothly, the presentation should
            # regularly check the display.closed property and exit
            # (before calling display.show) if that flag is True.
            while self._presentation is not None:
                self._presentation.join(timeout=1.0)
                if presentation.is_alive():
                    LOG.warning("GUI Display: background thread (presentation)"
                                " did not stop (opened=%s, active=%d)",
                                self.opened, self.active)
                else:
                    self._presentation = None

        event_loop = self._event_loop
        if isinstance(event_loop, threading.Thread):
            if event_loop is not threading.current_thread():
                event_loop.join()
            self._event_loop = None

        if self._on_close is not None:
            on_close = self._on_close
            self._on_close = None
            if isinstance(on_close, type) and issubclass(on_close, Exception):
                exception = on_close("Display was closed.")
                if self._entered > 0:
                    self._on_close = exception
                raise exception
            if isinstance(on_close, Callable):
                on_close()

    @property
    def opened(self) -> bool:
        """Check if this image :py:class:`Display` is opened, meaning
        the display window is shown. The `opened` state does not imply
        that an event loop is running (that may or may not be the case).
        To see if an event loop is running, check the :py:prop:`active`
        property.
        """
        return self._opened

    @property
    def active(self) -> bool:
        """Check if this :py:class:`Display` is active, meaning that an event
        loop is running.  If the display is :py:prop:`opened`, but not
        active, the caller is responsible for processing events, by
        regurlarly calling either :py:meth:`process_events` or
        :py:meth:`show` (which internally calls
        :py:meth:`process_events`).

        """
        return self._active and self._event_loop_is_running()

    @property
    def blocking(self) -> bool:
        """Blocking behaviour of this :py:class:`Display`.  `True` means that
        an event loop is run in the calling thread and execution of
        the program is blocked while showing the display, `False`
        means that the event loop is executed in a background thread
        or not at all (:py:prop:`active` is `False`), in which case
        event processing has to be triggered explicitly.
        """
        return self._blocking is True

    @property
    def closed(self) -> bool:
        """Check if this image :py:class:`Display` is closed, meaning
        that no window is shown (and no event loop is running).
        """
        return not self._opened

    #
    # methods to be implemented by subclasses
    #

    def _open(self) -> None:
        """Open the display window. The function is only called if
        the `Display` is not open yet, that is, it is not visible on
        screen.  After successfully opening the `Display`,
        the :py:prop:`opened` property should be `True`.

        This method essentially is a stub for implementing the
        :py:meth:`open` method in subclasses.
        """
        raise NotImplementedError(f"{type(self)} claims to be a Display, "
                                  "but does not implement an _open() method.")

    def _show(self, *args, **kwargs) -> None:
        """Show data on this display.

        This method is a stub for implementing the :py:meth:`show`
        method.  An implementation should only deal with showing the
        specific content.  It should not touch any of the event loop
        related aspects of the  :py:meth:`show` method.
        """
        raise NotImplementedError(f"{type(self).__name__} claims to "
                                  "be an Display, but does not implement "
                                  "the _show method.")

    def _close(self) -> None:
        """Close this `Display`.  This is a stub to be overwritten
        by subclasses to implement :py:meth:`close`.

        This method should only deal with removing the `Display` from
        screen.  All other aspects, like ending event loops, will be
        taken of before this method is called.
        """
        raise NotImplementedError(f"{type(self)} claims to be a Display, "
                                  "but does not implement an _close() method.")

    def _process_events(self) -> None:
        raise NotImplementedError(f"{type(self)} claims to be a Display, "
                                  "but does not implement "
                                  "_process_events().")

    def _run_event_loop(self) -> None:
        if self.blocking is True:
            self._run_blocking_event_loop()
        elif self.blocking is False:
            self._run_nonblocking_event_loop()

    def _dummy_event_loop(self, timeout: float = None) -> None:
        # pylint: disable=broad-except
        interval = 0.1

        start = time.time()
        try:
            LOG.info("Display: start dummy event loop (closed=%s)",
                     self.closed)
            while (not self.closed and
                   (timeout is None or time.time() < start + timeout)):
                self._process_events()
                time.sleep(interval)
        except BaseException as exception:
            LOG.error("Unhandled exception in event loop")
            handle_exception(exception)
        finally:
            LOG.info("Display: ended dummy event loop (closed=%s).",
                     self.closed)
            self._event_loop = None
            self.close()

    def _run_blocking_event_loop(self, timeout: float = None) -> None:
        self._event_loop = threading.current_thread()
        self._dummy_event_loop(timeout)

    def _run_nonblocking_event_loop(self) -> None:
        """Start a dummy event loop. This event loop will run in the
        background and regularly trigger event processing. This may be
        slightly less responsive than running the official event loop,
        but it has the advantage that this can be done from a background
        Thread, allowing to return the main thread to the caller.
        In other words: this function is intended to realize a non-blocking
        image display with responsive image window.

        FIXME[todo]: check how this behaves under heavy load (GPU computation)
        and if in case of problems, resorting to a QThread would improve
        the situation.
        """
        if self._event_loop_is_running():
            raise RuntimeError("Only one event loop is allowed.")
        self._event_loop = \
            threading.Thread(target=self._nonblocking_event_loop)
        self._event_loop.start()

    def _nonblocking_event_loop(self) -> None:
        self._dummy_event_loop()

    def _event_loop_is_running(self) -> bool:
        """Check if an event loop is currently running.
        """
        return self._event_loop is not None

    def _run_background_thread(self, task: Callable) -> None:
        self._background_thread = \
            Thread(self._background_wrapper, name='Display-background',
                   args=(task,))
        self._background_thread.start()

    def _background_wrapper(self, task: Callable) -> None:
        task()
        self.stop()

    #
    # FIXME[refactor]
    #

    # FIXME[refactor]: the presenter can be passed as argument `blocking`
    # to the `show` method
    def present(self, presenter, args=(), kwargs={}) -> None:
        # pylint: disable=dangerous-default-value
        """Run the given presenter in a background thread while
        executing the GUI event loop in the calling thread (which
        by some GUI library is supposed to be the main thread).

        The presenter will get the display as its first argument,
        and `args`, `kwargs` as additional arguments. The presenter
        may update the display by calling the :py:meth:`show` method.
        The presenter should observe the display's `closed` property
        and finish presentation once it is set to `True`.

        Arguments
        ---------
        presenter:
            A function expecting a display object as first argument
            and `args`, and `kwargs` as additional arguments.
        """
        def target() -> None:
            # pylint: disable=broad-except
            LOG.info("Display[background]: calling presenter")
            try:
                presenter(self, *args, **kwargs)
            except BaseException as exception:
                LOG.error("Unhandled exception in presentation.")
                handle_exception(exception)
            finally:
                self.close()

        with self:
            LOG.info("Display[main]: Starting presentation")
            self._presentation = threading.Thread(target=target)
            self._presentation.start()
            self._run_blocking_event_loop()


class View:
    """Abstract base class for GUI elements that allow to view data.
    Views can be displayed in a suitable :py:class:`Display`.
    """


class SimpleDisplay(Display, ABC):
    """A `SimpleDisplay` is a base class for simple displays,
    usually only displaying a single component (:py:class:`View`).
    """
    View: type = View

    _view: View

    def __init__(self, view: Optional[View] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.view = self.View() if view is None else view

    @property
    def view(self) -> View:
        """The (only) :py:class:`View` employed in this `SimpleDisplay`
        """
        return self._view

    @view.setter
    def view(self, view: Optional[View]) -> None:
        if not isinstance(view, self.View):
            raise TypeError(f"View shoulb be of type {self.View}.")
        self._view = view
