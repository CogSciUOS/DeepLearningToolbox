"""Tests for the :py:class:`Tool` class.
"""

# standard imports
from unittest import TestCase
from typing import Tuple

# third party imports
import numpy as np

# toolbox imports
from dltb.base.data import Data
from dltb.tool import Tool


class SingleTestTool(Tool):
    """
    """
    
    def _process_single(array: np.ndarray) -> Tuple[float, bool]:
        """Process a single piece of data.
        """
        return array.sum(), False


class BatchTestTool(Tool):
    """
    """

    def _process_batch(batch: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process a batch of data.
        """
        return batch.sum(axis=1), True


class DualTestTool(SingleTestTool, BatchTestTool):
    """A tool implementing both, single and batch processing.
    """


class TestTool(TestCase):
    """Tests for the :py:class:`Tool` class.
    """

    def test_single_tool_01(self) -> None:
        """Call the :py:class:`SingleTestTool` with a single argument.
        """
        tool = SingleTestTool()
        result, batch = tool([1, 2, 3])
        self.assertEqual(result, 6)
        self.assertIsFalse(batch)

    def test_single_tool_02(self) -> None:
        """Call the :py:class:`SingleTestTool` with a single
        :py:class:`Data` argument.
        """
        data = Data(array=[1, 2, 3], batch=False)
        tool, batch = SingleTestTool(data)
        result = tool(data)
        self.assertIsInstance(result, int)
        self.assertIsFalse(batch)

    def test_batch_tool_01(self) -> None:
        """Call the :py:class:`SingleTestTool` with a single
        (non-batch) argument.
        """
        tool = BatchTestTool()
        result, batch = tool([1, 2, 3])
        self.assertEqual(result, 6)
        self.assertIsTrue(batch)

    def test_batch_tool_02(self) -> None:
        """Call the :py:class:`BatchTestTool` with a single (non-batch)
        :py:class:`Data` argument.
        """
        data = Data(array=[1, 2, 3], batch=False)
        tool = BatchTestTool()
        result, batch = tool(data)
        self.assertEqual(result, 6)
        self.assertIsTrue(batch)
