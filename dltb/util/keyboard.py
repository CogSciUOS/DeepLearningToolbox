"""Support for keyboard interaction.

Python has only little support for direct keyboard interactions.  Most
functions are operating system dependent or provided by third party
libraries (like `getch`).  The main goal of this module is to provide
a uniform API that can be used to detect individual keystrokes.

"""

# standard imports
from typing import Type, Optional, Iterable
from types import TracebackType
import sys
import threading
import select


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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.thread = None
        self.key_pressed = None
        self.capture_loop = False

    def __call__(self, iterable) -> Iterable:
        with self:
            for item in iterable:
                yield item
                if self:
                    break

    def __enter__(self) -> 'KeyPress':
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 exc_traceback: Optional[TracebackType]) -> bool:
        thread = self.thread
        if thread is not None:
            self.stop_capture()
            thread.join()
        print("Leaving the context manager")

    def __bool__(self) -> bool:
        return self.key_pressed is not None

    def start_capture(self) -> None:
        """Start a key capturing process.
        """
        if self.thread is not None:
            raise RuntimeError("KeyPress is already running")
        self.thread = threading.Thread(target=self.capture_key, args=(),
                                       name='key_capture_thread', daemon=True)
        self.thread.start()

    def stop_capture(self) -> None:
        """Stop the key capturing process.
        """
        # end the background thread
        # print("Please hit the enter key to finish the KeyPress Manager")
        print("Gracefully exiting the capture loop")
        self.capture_loop = False

    def run_capture_key(self) -> None:
        """Run the :py:meth:`capture_key` method and clean up, once it
        finishes.

        """
        self.capture_key()
        self.thread = None
        print("Finished capturing keys.")

    def capture_key(self) -> None:
        """Run a loop to capture a key (to be implemented by subclasses).
        Once a key is pressed, this method should set the property
        :py:prop:`key_pressed` and stop.
        """


class DummyKeyboardObserver(KeyboardObserver):
    """A :py:class:`KeyboardObserver` that uses python's `input` function
    to check for keys.  This will only receive a message, once the return
    key is pressed.
    """

    def capture_key(self) -> None:
        """Run a loop to capture a key.
        """
        self.key_pressed = input()
        self.thread = None

    def stop_capture(self) -> None:
        """Stop the key capturing process.
        """
        # end the background thread
        print("Please hit the enter key to finish the KeyPress Manager")


class LoopKeyboardObserver(KeyboardObserver):
    """An auxilary :py:class:`KeyboardObserver` running a loop to
    regularly check if a key was pressed.

    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.capture_loop = False

    def capture_key(self) -> None:
        """Run a loop to capture a key.
        """
        self.capture_loop = True
        while self.capture_loop and not self.key_pressed:
            self.check_for_key()

    def check_for_key(self, timeout=.1) -> None:
        """Check if a key was pressed (to be implemented by subclasses).
        """

    def stop_capture(self):
        """Stop the key capturing process.
        """
        print("Gracefully exiting the capture loop")
        self.capture_loop = False


class SelectKeyboardObserver(LoopKeyboardObserver):
    """A :py:class:`KeyboardObserver` using the (standard) `select` module
    to check if a key was pressed.
    """

    def check_for_key(self, timeout=.1) -> None:
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

    def check_for_key(self, timeout=.1) -> None:
        """Run a loop to capture a key.
        """
        # from getch impor getch
        self.key_pressed = None  # getch()
