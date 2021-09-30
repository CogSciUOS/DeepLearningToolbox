"""Support for keyboard interaction.

Python has only little support for direct keyboard interactions.  Most
functions are operating system dependent or provided by third party
libraries (like `getch`).  The main goal of this module is to provide
a uniform API that can be used to detect individual keystrokes.

"""

# standard imports
from typing import Type, Optional, Iterable
from types import TracebackType
from threading import Thread
import sys
import select
import logging

# logging
LOG = logging.getLogger(__name__)


class KeyboardObserver:
    """Functionality for checking key presses.  This may be used in
    loops to stop the loop when a key is pressed.

    .. highlight:: python
    .. code-block:: python

        with KeyboardObserver() as key_pressed:
            while not key_pressed:
                ...

    .. highlight:: python
    .. code-block:: python
        stop_on_key = KeyboardObserver()
        for i in stop_on_key(range(1000000)):
            print(i)
    """
    _thread: Optional[Thread] = None
    key_pressed: Optional[str] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]
        self._thread = None
        self.key_pressed = None

    def __call__(self, iterable) -> Iterable:
        with self:
            for item in iterable:
                yield item
                if self:
                    break

    def __enter__(self) -> 'KeyboardObserver':
        LOG.info("Entering the KeyboardObserver context manager")
        self.start_capture()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 exc_traceback: Optional[TracebackType]) -> None:  # bool
        # Remark: type checker prefers return type None or Literal[False] over
        # bool to signal that the context manager will not swallow exception
        self.stop_capture()
        LOG.info("Leaving the KeyboardObserver context manager")
        # return False  # no exception handling was done in the __exit__ method

    def __bool__(self) -> bool:
        return self.key_pressed is not None

    def start_capture(self) -> None:
        """Start a key capturing process.
        """
        LOG.debug("KeyboardObserver: starting a capturing thread")

        if self._thread is not None:
            raise RuntimeError("KeyPress is already running")
        self._thread = Thread(target=self._run_capture_key, args=(),
                              name='key_capture_thread', daemon=True)
        self._thread.start()

    def stop_capture(self) -> None:
        """Stop the key capturing process.
        """
        thread = self._thread
        if thread is not None:
            LOG.debug("KeyboardObserver: stopping the capturing thread")
            self._stop_capture()
            thread.join()
            self._thread = None
            LOG.debug("KeyboardObserver: capturing thread ended.")
        else:
            LOG.debug("KeyboardObserver: no capturing thread is running")

    def _run_capture_key(self) -> None:
        """Run the :py:meth:`capture_key` method and clean up, once it
        finishes.

        """
        self._capture_key()
        self._thread = None
        LOG.info("Finished capturing keys with KeyboardObserver.")

    def _capture_key(self) -> None:
        """Run a loop to capture a key (to be implemented by subclasses).
        Once a key is pressed, this method should set the property
        :py:prop:`key_pressed` and stop.
        """
        # to be implemented by subclasses

    def _stop_capture(self) -> None:
        """Stop the key capturing process (to be implemented by sublcasses).
        """
        # to be implemented by subclasses


class DummyKeyboardObserver(KeyboardObserver):
    """A :py:class:`KeyboardObserver` that uses python's `input` function
    to check for keys.  This will only receive a message, once the return
    key is pressed.
    """

    def _capture_key(self) -> None:
        """Run a loop to capture a key.
        """
        self.key_pressed = input()

    def _stop_capture(self) -> None:
        """Stop the key capturing process.
        """
        # end key capturing
        print("Please hit the enter key to finish the KeyPress Manager")


class LoopKeyboardObserver(KeyboardObserver):
    """An auxilary :py:class:`KeyboardObserver` running a loop to
    regularly check if a key was pressed.

    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.capture_loop = False

    def _capture_key(self) -> None:
        """Run a loop to capture a key.
        """
        self.capture_loop = True
        while self.capture_loop and not self.key_pressed:
            self._check_for_key()

    def _check_for_key(self, timeout=.1) -> None:
        """Check if a key was pressed.
        """
        # to be implemented by subclasses

    def _stop_capture(self):
        """Stop the key capturing process.
        """
        self.capture_loop = False


class SelectKeyboardObserver(LoopKeyboardObserver):
    """A :py:class:`KeyboardObserver` using the (standard) `select` module
    to check if a key was pressed.
    """

    def _check_for_key(self, timeout=.1) -> None:
        """Check if a key was pressed.
        """
        # in_state is either an empty list (if no input is available)
        # of a list containing only sys.stdin (if input is available).
        # Input will only be available upon pressing return!
        in_state, _o, _e = select.select([sys.stdin], [], [], timeout)
        if in_state:
            self.key_pressed = sys.stdin.readline().strip()
            print("You said", self.key_pressed)


class GetchKeyboardObserver(LoopKeyboardObserver):
    """A :py:class:`KeyboardObserver` using the (third party) `getch`
    module to check if a key was pressed.
    """

    def _check_for_key(self, timeout=.1) -> None:
        """Run a loop to capture a key.
        """
        # from getch impor getch
        self.key_pressed = None  # getch()
