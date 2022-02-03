"""Plotting utilities
"""

# standard imports
from typing import Optional, Union, Callable

# third-party imports
import numpy as np

# toolbox imports
from ..base.implementation import Implementable


class Plotter(Implementable):
    """A `Plotter` can plot.
    """

    def save(self, filename: str) -> None:
        """Save the current plot.
        """


class TilingPlotter(Plotter, Implementable):
    """Plot a tile from a batch of images.
    """

    def plot_tiling(self, images: np.ndarray,
                    rows: Optional[int] = None,
                    columns: Optional[int] = None) -> None:
        """Tiling plot.
        """


class Scatter2dPlotter(Plotter, Implementable):
    """Plot a 2D scatter plot.
    """

    def plot_scatter_2d(self, data: Optional[np.ndarray] = None,
                        xvals: Optional[np.ndarray] = None,
                        yvals: Optional[np.ndarray] = None,
                        labels: Optional[np.ndarray] = None) -> None:
        """Scatter plot.
        """


class Display(Implementable):
    """A graphical user interface to display Plotters.
    """

    def show(self) -> None:
        """Show the display.
        """


Plotterlike = Union[Plotter, Display]

def plotting(plotter_class: type) -> Callable:
    """A decorator to mark a function as doing some plotting.  Such
    a function uses a plotter being a subclass of `plotter_class`, which
    is provided by an argument named 'plotter'.
    """
    def decorator(function) -> Callable:
        def wrapper(*args, plotter: Optional[Plotterlike] = None, **kwargs):
            if isinstance(plotter, plotter_class):
                the_plotter = plotter
            elif isinstance(plotter, Display):
                the_plotter = plotter_class(display=plotter)
            elif plotter is None:
                plotter_implementation = plotter_class.get_implementation()
                display = plotter_implementation.Display()
                the_plotter = plotter_implementation(display=display)
            else:
                raise TypeError("Cannot use plotter of type "
                                f"{type(plotter).__name__} as "
                                f"{plotter_class.__name__}")

            function(*args, plotter=the_plotter, **kwargs)
            if plotter is None:  # stand-alone mode
                display.show()  # block until display is closed
                # FIXME[todo]: alternative would be to not show and/or
                # return display and maybe also the_plotter
        return wrapper
    return decorator
