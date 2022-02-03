"""Testsuite for the `dltb.base.run` module.
"""
# pylint false positive for decorator keyword arguments:
# https://github.com/PyCQA/pylint/issues/258
# supposed to be fixed in pylint 2.13.0
# https://github.com/PyCQA/pylint/milestone/49
#
# pylint: disable=unexpected-keyword-arg

# standard imports
from threading import Lock, current_thread
import unittest

# toolbox imports
from dltb.base.run import runnable, run_synchronous, run_asynchronous


class Mockup:
    """A mockup class to test the `@runnable` decorator.
    """

    def __init__(self) -> None:
        self.lock = Lock()
        self.result = None

    def method1(self) -> str:
        """A dummy method, returning the name of the current thread.

        """
        with self.lock:
            name = current_thread().name
        return name

    @runnable
    def method2(self) -> str:
        """A second implementation of the method marked as `@runnable`.
        """
        with self.lock:
            name = current_thread().name
        return name

    def process_result(self, result: str) -> str:
        """Callback to process results from method invocation.
        """
        self.result = result


class TestRun(unittest.TestCase):
    """Tests for the `@runnable` decorator.
    """

    def test_run_synchronous(self):
        """Test the :py:func:`run_synchronous` function.
        """
        mockup = Mockup()
        result = run_synchronous(mockup.method1)
        self.assertEqual(current_thread().name, result)

    def test_run_synchronous_callback(self):
        """Test the :py:func:`run_synchronous` function with callback.
        """
        mockup = Mockup()
        run_synchronous(mockup.method1, run_callback=mockup.process_result)
        self.assertEqual(current_thread().name, mockup.result)

    def test_run_asynchronous(self):
        """Test the :py:func:`run_asynchronous`.
        """
        mockup = Mockup()
        thread = run_asynchronous(mockup.method1)
        thread.join()
        self.assertEqual(None, mockup.result)

    def test_run_asynchronous_callback(self):
        """Test the :py:func:`run_asynchronous` function with callback.
        """
        mockup = Mockup()
        thread = run_asynchronous(mockup.method1,
                                  run_callback=mockup.process_result)
        self.assertEqual(thread.name, mockup.result)

    def test_runnable_false(self):
        """Test the `@runnable` decorator with argument `run=False`.
        """
        mockup = Mockup()
        result = mockup.method2(run=False)
        self.assertEqual(current_thread().name, result)

    def test_runnable_true(self):
        """Test the `@runnable` decorator with argument `run=True`.
        """
        mockup = Mockup()
        thread = mockup.method2(run=True)
        thread.join()
        self.assertEqual(None, mockup.result)

    def test_runnable_callback(self):
        """Test the `@runnable` decorator with callback function.
        """
        mockup = Mockup()
        thread = mockup.method2(run_callback=mockup.process_result)
        thread.join()
        self.assertEqual(thread.name, mockup.result)

    def test_runnable_name(self):
        """Test the `@runnable` decorator with `run_name` argument.
        """
        mockup = Mockup()
        name = 'test'
        thread = mockup.method2(run_name=name,
                                run_callback=mockup.process_result)
        thread.join()
        self.assertEqual(name, thread.name)
        self.assertEqual(name, mockup.result)
