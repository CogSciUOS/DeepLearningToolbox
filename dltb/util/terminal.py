"""An abstract interface for a terminal that allows for text output and
simple input. The default implementation assumes a simple text terminal.
Extensions may provide implementations to be used in more advanced
settings, like graphical user interfaces.
"""

class Terminal:
    """A simple text-based terminal.
    """

    class Bformat:
        # pylint: disable=too-few-public-methods
        """Escape codes for color output.
        """
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        ENDC = '\033[0m'

    class Bstatus:
        # pylint: disable=too-few-public-methods
        """Symbols for semantic markup.
        """
        HEADER = '\033[95m'
        OK = '\033[92m'  # GREEN
        OK2 = '\033[94m'  # BLUE
        WARNING = '\033[93m'  # YELLOW
        FAIL = '\033[91m'  # RED

    class Markup:
        # pylint: disable=too-few-public-methods
        """Symbols for semantic markup.
        """
        EMPHASIZE = '\033[94m'  # BLUE

    def form(self, text: str, color: str = '') -> str:
        """Format a text in a given color.
        """
        return color + text + self.Bformat.ENDC

    def status(self, text: str, status: str = None) -> str:
        """Format a text with a given status code.
        """
        return self.form(text, status and
                         getattr(self.Bstatus, status.upper()))

    def markup(self, text: str, markup: str = None) -> str:
        """Format a text with a given Markup code.
        """
        return self.form(text, markup and
                         getattr(self.Markup, markup.upper()))

    def output(self, message: str) -> None:
        """
        """
        print(message)


DEFAULT_TERMINAL = Terminal()
