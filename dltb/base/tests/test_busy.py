"""Testsuite for the `dltb.base.run` module.
"""
# pylint false positive for decorator keyword arguments:
# https://github.com/PyCQA/pylint/issues/258
# supposed to be fixed in pylint 2.13.0
# https://github.com/PyCQA/pylint/milestone/49
#
# pylint: disable=unexpected-keyword-arg

# standard imports
from threading import Thread, Lock, current_thread
import unittest

# toolbox imports
from dltb.base.busy import busy, BusyObject


BUSY_MESSAGE = "being busy"

class Mockup(BusyObject):
    """A mockup class to test the `@busy` decorator.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lock = Lock()
        self.result = None

    @busy(BUSY_MESSAGE)
    def busy_method(self) -> str:
        """A simple method waiting to get unlocked.
        """
        with self.lock:
            name = current_thread().name
        return name

    def process_result(self, result: str) -> str:
        """Callback to process results from method invocation.
        """
        self.result = result


class TestRun(unittest.TestCase):
    """Tests for the :py:class:`BusyObject` class.
    """

    def test_run_synchronous(self):
        """Test the busy method (synchrnounous).
        """
        mockup = Mockup()
        self.assertFalse(mockup.busy)
        name = mockup.busy_method(run=False)
        self.assertIsInstance(name, str)
        self.assertEqual(name, current_thread().name)

    def test_run_asynchronous(self):
        """Test the busy method (asynchrnounous).
        """
        mockup = Mockup()
        self.assertFalse(mockup.busy)

        with mockup.lock:
            thread = mockup.busy_method(run=True)
            self.assertIsInstance(thread, Thread)
            self.assertTrue(mockup.busy)
            self.assertEqual(mockup.busy_message, BUSY_MESSAGE)
        thread.join()
        self.assertFalse(mockup.busy)
        self.assertEqual(mockup.busy_message, '')
