"""API for separators (like speech separators).
"""

# standard imports
from typing import Tuple

# toolbox imports
from ..base.sound import Sound


class SpeechSeparator:
    """Abstract Interface for implementing speech separators,
    providing some convenience functions.
    """

    def __call__(self, mixed: Sound) -> Tuple[Sound, ...]:
        return self._separate_files(mixed)

    def _separate_files(self, mixed: Sound) -> Tuple[Sound, ...]:
        """Separate the given Sounds.
        """
