"""CUDA related code.
"""


class CUDA:
    """Interface to access CUDA functionality.
    """

    @staticmethod
    def is_avalable() -> bool:
        """Check if CUDA support is available.
        """
        return True  # FIXME[todo]

    @staticmethod
    def number_of_gpus() -> int:
        """Determine how many GPUs are available.
        """
        return 1  # FIXME[todo]
