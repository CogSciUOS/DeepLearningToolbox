"""
"""

from typing import Iterable

from .tool import Tool


class Evaluator:
    """An :py:class:`Evaluator` can evaluate a (specific type) of model.
    """

    def __init__(self, tool: Tool = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tool = tool

    def reset(self) -> None:
        """Reset the evaluation process.
        """

    def evaluate(self, data: Iterable) -> None:
        """Evaluate a sequence of data.  The internal evaluation scores
        will be updated.
        """
