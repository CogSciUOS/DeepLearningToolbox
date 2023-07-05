"""

from dltb.thirdparty.matplotlib import MatplotlibPlotter
plotter = MatplotlibPlotter()
plotter.show()

"""

# standard imports
from typing import Callable, Optional, Any
import logging

# third party imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

# toolbox imports
from dltb.base.gui import Display, SimpleDisplay, View
from dltb.base.image import Image, Imagelike
from dltb.base.image import ImageReader, ImageWriter, ImageDisplay
from dltb.base.run import MainThreader, main_thread_only, main_thread_guard
from dltb.util.plot import Display as DisplayBase
from dltb.util.plot import TilingPlotter, Scatter2dPlotter

# logging
LOG = logging.getLogger(__name__)

# FIXME[old/design]: what are all this classes used for:
#  - class MplDisplay(DisplayBase, MainThreader):


class MplDisplay(Display, MainThreader):
    """A display for showing a matplotlib figure.


    Threading
    ---------

    "Most GUI frameworks require that all updates to the screen, and
    hence their main event loop, run on the main thread. This makes
    pushing periodic updates of a plot to a background thread
    impossible. Although it seems backwards, it is typically easier to
    push your computations to a background thread and periodically
    update the figure on the main thread." [1]


    Matplotlib event loop
    ---------------------

    Matplotlib can be in different modes, with the current mode
    indicated by `rcParams['interactive']`. The value `True` indicates
    that matplotlib is in interactive mode (the function
    `matplotlib.is_interactive`, and also the alias
    `matplotlib.pyplot.isinteractive` will return `True`).  In
    interactive mode, the figure is redrawn after every plotting
    command.  The interactive mode can be turned on (or off) by calling
    `matplotlib.interactive(True)` (or `False`, respectively).

    The convenience function `matplotlib.pyplot.ion` turns on
    interactive mode and in addtion install a so called
    Read–eval–print-loop (REPL) hook. Such a hook can automatically
    redraw any stall figure automatically when control is returned
    to that loop.
    
    [1] https://matplotlib.org/stable/users/explain/interactive_guide.html#threading

    """

    # FIXME[todo]: handle th different cases: GUI/IPython/standalone
    #
    # Matplotlib: :figure.show():
    #
    # If using a GUI backend with pyplot, display the figure window.
    #
    # If the figure was not created using `~.pyplot.figure`, it will lack
    # a `~.backend_bases.FigureManagerBase`, and this method will raise an
    # AttributeError.
    #
    # This does not manage an GUI event loop. Consequently, the figure
    # may only be shown briefly or not shown at all if you or your
    # environment are not managing an event loop.
    #
    # Proper use cases for `.Figure.show` include running this from a
    # GUI application or an IPython shell.
    #
    # If you're running a pure python shell or executing a non-GUI
    # python script, you should use `matplotlib.pyplot.show` instead,
    # which takes care of managing the event loop for you.
    #
    
    
    running_number: int = 101

    _figure: Figure = None
    _close_id = None

    default_figsize = (8, 6)  # FIXME[hack]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._update_values = None

    @property
    def figure(self) -> Optional[Figure]:
        """The Matplotlib `Figure` used by this `MplDisplay`.
        Can be `None` if the display is currently inactive.
        """
        return self._figure   

    @figure.setter
    def figure(self, figure: Optional[Figure]) -> None:
        if figure is not self._figure:
            self._set_figure(figure)

    def _set_figure(self, figure: Optional[Figure]) -> None:
        if self._figure is not None:
            self._figure.canvas.mpl_disconnect(self._close_id)
        self._figure = figure
        if self._figure is not None:
            self._close_id = \
                self._figure.canvas.mpl_connect('close_event',
                                                self._handle_close)

    def _create_figure(self, number: int) -> None:
        return plt.figure(figsize=self.default_figsize, num=number)

    # FIXME[old/todo]: integrate this into the general dltb.base.gui logic
    @main_thread_only
    def show_old(self, on_close: Callable = None) -> None:
        """
        """
        self._open()

        # FIXME[todo]: maybe we can get a better integration with the
        # matplotlib GUI event loop logic.  In principle it should be
        # possible to continously run the event loop (e.g. by calling
        # canvas.start_event_loop() or using show(block=True) and then
        # interrupt this event loop only if new data are available.
        # This would relief us from running our own (suboptimal) event
        # loop.
        # https://matplotlib.org/stable/users/explain/interactive_guide.html

        # plt.ion()
        # plt.show()
        self._figure.show()
        with self.main_thread_loop():
            try:
                while plt.fignum_exists(self._figure.number):
                    self.perform_main_thread_activities()
                    # draw_idle catches KeyboardInterrupt and hence cannot
                    # be interrupted. Therefor use draw() here
                    # self._figure.canvas.draw_idle()
                    self._figure.canvas.draw()
                    self._figure.canvas.flush_events()
                    # plt.pause(0.01)
            except KeyboardInterrupt:
                LOG.warning("KeyboardInterrupt in event loop")
                plt.close(self._figure)
        if on_close is not None:
            on_close()

    def save(self, filename: Optional[str] = None) -> None:
        """Save the current content of the display to a file.
        """
        path = os.path.join("./plot", filename + ".png")
        plt.savefig(path)
        LOG.info(f"Saving figure to '{path}'")


    def show_figure(self, figure) -> None:
        """Show the given MatPlotLib figure.
        """
        self._figure = figure
        # plt.ion() runs:
        #  1. matplotlib.interactive(True)
        #  2. install_repl_displayhook())
        plt.ion()

        # plt.show()
        plt.show()

    def hide_figure(self) -> None:
        """
        """

    def _handle_close(self, evt):
        LOG.info('Matplotlib Display: received a close event.')
        self.close()

    #
    # implementation of dltb.base.gui.Display methods
    #

    def _open(self) -> None:
        """Open the display window. The function is only called if
        the `Display` is not yet opened.
        """
        if self._figure is None:
            self.figure = \
                self._create_figure(number=MplDisplay.running_number)
            MplDisplay.running_number += 1
        elif not plt.fignum_exists(self._figure.number): # figure is closed
            # https://stackoverflow.com/a/31731945
            dummy = plt.figure(num=self._figure.number)
            new_manager = dummy.canvas.manager
            new_manager.canvas.figure = self._figure
            self._figure.set_canvas(new_manager.canvas)
        self._figure.show()

    # def _show(self, image: np.ndarray, wait_for_key: bool = False,
    #           timeout: float = None, **kwargs) -> None:
    def _show(self, *args, **kwargs) -> None:
        title = kwargs.get('title', "Image")
        self._figure.canvas.set_window_title(title)
        self._figure.canvas.draw()
        self._figure.suptitle("MatPlotLib")

    def _close(self) -> None:
        """Close the MatPlotLib `Display`.
        """
        plt.close(self._figure)

    def _process_events(self) -> None:
        # FIXME[todo]: no idea if this really processes event - some
        # more research is necessary here!
        self._figure.canvas.draw()
        plt.draw_all()
        plt.pause(0.01)

    def _run_nonblocking_event_loop(self) -> None:
        # Under IPython: self._figure.show(), the IPython shell will manage
        # the event loop:
        plt.ion()
        # self._figure.show()
        plt.show(block=False)

    def _run_blocking_event_loop(self, timeout: float = None) -> None:

        # matlplotlib.pyplot.show(block: bool):
        #
        # In non-interactive mode, *block* defaults to True.  All
        # figures will display and show will not return until all
        # windows are closed.  If there are no figures, return
        # immediately.
        # 
        # In interactive mode *block* defaults to False.  This will
        # ensure that all of the figures are shown and this function
        # immediately returns.
        #
        # Parameters
        # ----------
        # block : bool, optional
        # 
        #     If `True` block and run the GUI main loop until all windows
        #     are closed.
        # 
        #     If `False` ensure that all windows are displayed and return
        #     immediately.  In this case, you are responsible for ensuring
        #     that the event loop is running to have responsive figures.
        #
        # See Also
        # --------
        # ion : enable interactive mode
        # ioff : disable interactive mode
        # plt.ioff()
        plt.ion()
        plt.show(block=True)

    def _event_loop_is_running(self) -> bool:
        """Check if an event loop is currently running.
        """
        return self._active


class MplPlotter(View):
    """A Matplitlib Plotter can plot on a Matplpotlib 'Axes'.
    """
    _axes = None

    # FIXME[concept/old]: a Display should now what View(s) it can display,
    # but (why) does a View need to know in which display it can be
    # displayed? -> maybe to automatically create a suitable Display?
    Display = MplDisplay

    def __init__(self, display: Optional[MplDisplay] = None,
                 axes: Optional[Axes] = None, **kwargs) -> None:
        super().__init__(**kwargs)

        # FIXME[old]: is the display really needed?
        # the display is responsible for displaying data (event loop, etc.)
        self._display = display

        # the axes is used by an MplPlotter to do the actual drawing.
        if display is None:
            self.axes = axes
        else:
            if axes is not None and axes is not display.axes:
                ValueError("Provided axis does not belong to display.")
            self.axes = display.axes

    @property
    def axes(self) -> Axes:
        return self._axes

    @axes.setter
    def axes(self, axes: Axes) -> None:
        if axes is self._axes:
            return
        if self._axes is not None:
            self._release_axes()
        self._axes = axes
        if axes is not None:
            self._axes.clear()
            self._init_axes()
        self._figure = None if axes is None else axes.figure

    def _init_axes(self) -> None:
        pass

    def _release_axes(self) -> None:
        self._axes.clear()


class MplSimpleDisplay(MplDisplay, SimpleDisplay):
    """A `MplSimpleDisplay` is the abstract base class for simple
    matplotlib displays.  It keeps a `Figure` with a single `Axes` object
    accessible as :py:prop:`axes`.
    """

    _figure: Optional[Figure] = None
    _axes: Optional[Axes] = None

    def __init__(self, view: Optional[View] = None,
                 plotter: Optional[MplPlotter] = None, **kwargs) -> None:
        """
        """
        if plotter is not None:
            view = plotter
        super().__init__(view=view, **kwargs)

    @property
    def axes(self) -> Axes:
        """The single `Axes` object of this :py:class:`MplSimpleDisplay`.
        """
        return self._axes

    @property
    def plotter(self) -> MplPlotter:
        """The Matplotlib plotter used in this :py:class:`MplSimpleDisplay`.
        """
        return self.view

    @plotter.setter
    def plotter(self, plotter: MplPlotter) -> None:
        if self.plotter is not None:
            self.plotter.axes = None
        self.view = plotter
        if self.plotter is not None:
            self.plotter.axes = self.axes

    def _create_figure(self, number: int) -> None:
        figure, self._axes = \
            plt.subplots(1, 1, figsize=self.default_figsize, num=number)
        if self.view is not None:
            self.view.axes = self._axes     
        return figure

    def _open_old(self) -> None:
        if self._figure is not None:
            # self._figure.canvas.draw()
            self._figure.show()
            return
        if self._blocking is not True:
            plt.ion()  # enable interactive mode
        #if self._blocking is not True:
        #    self._figure.show(block=False)  # matplotlib 1.1


class MplTilingPlotter(TilingPlotter, MplPlotter):
    """Matplotlib implementation of the :py:class:`TilingPlotter`.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._imshow = None

    def _init_axes(self) -> None:
        super()._init_axes()
        self._imshow = None

    def _release_axes(self) -> None:
        self._imshow = None
        super()._release_axes()

    def plot_tiling(self, images: np.ndarray,
                    rows: Optional[int] = None,
                    columns: Optional[int] = None) -> None:
        """Display a batch of images in a 2D grid.

        Arguments
        ---------
        images:
            A batch of images, either given as a one or two dimensional
            array of images.  If this is a two-dimensional array,
            this 2D shape will be taken as the tiling size (rows, columns).
        
        """
        # ensure that images has shape
        # (rows, columns, img_height, img_width, colors)
        if images.shape[-1] not in (1, 3):
            images = np.reshape(images, images.shape + (1, ))
        if images.ndim == 5:  # (rows, columns, height, width, color)
            rows, columns = images.shape[0:2]
        elif images.ndim == 4:
            if len(images) != rows*columns:
                raise ValueError(f"Rows ({rows}) * columns ({columns}) should"
                                 f" be the number of images ({len(images)})")
            images = np.reshape(images, (rows, columns) + images.shape[1:])

        self._plot(images)

    def _plot(self, images: np.ndarray) -> None:
        # FIXME[hack]: deal with synchronization problems: don't plot if
        # plotter has been deactivated ... find a better solution!
        figure = self._figure
        axes = self._axes
        if figure is None or axes is None:
            return

        rows, columns = images.shape[:2]
        canvas = self._tiling_image(images)
        if self._imshow is None:
            figure.set_figwidth(int(columns*0.8))
            figure.set_figheight(int(rows*0.8))

            cmap = 'gray' if canvas.ndim == 2 else None
            self._imshow = axes.imshow(canvas, cmap=cmap)
        else:
            self._imshow.set_data(canvas)

    @staticmethod
    def _tiling_image(images: np.ndarray) -> np.ndarray:
        rows, columns, height, width, channels = images.shape
        if channels == 1:
            # "MNIST": (batch, height, width)
            canvas = np.empty((rows * height, columns * width))
            for row, column in np.ndindex((rows, columns)):
                canvas[height*row: height*(row+1),
                       width*column: width*(column+1)] = \
                    np.reshape(images[row, column], (height, width))
        else:
            # "Cifar": (batch, height, width, color)
            canvas = np.empty((rows * height, columns * width, 3))
            for row, column in np.ndindex((rows, columns)):
                canvas[height*row: height*(row+1),
                       width*column: width*(column+1), :] = \
                    images[row, column]
        return canvas


class MplScatter2dPlotter(Scatter2dPlotter, MplPlotter):
    """Matplotlib implementation of the :py:class:`Scatter2dPlotter`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._scatter_2d = None
        self._divider = None
        self._colorbar = None
        self._colormap = None
        self._cax = None
        self._original_position = None

    def _init_axes(self) -> None:
        super()._init_axes()
        self._imshow = None
        self._original_position = self._axes.get_position()

    def _release_axes(self) -> None:
        self._imshow = None
        if self._cax is not None:
            self._cax.remove()
            self._cax = None
        self._divider = None
        self._colorbar = None
        self._colormap = None
        # FIXME[bug]: find out how to reset the axes position so that
        # it uses the original (full) size
        # self._axes.set_position(self._original_position)
        # self._axes.set_position([0., 0., 1., 1.])
        super()._release_axes()

    def plot_scatter_2d(self, data: Optional[np.ndarray] = None,
                        xvals: Optional[np.ndarray] = None,
                        yvals: Optional[np.ndarray] = None,
                        labels: Optional[np.ndarray] = None) -> None:
        """Create a 2D scatter plot.

        Arguments
        ---------
        xvals, yvals:
            Two arrays of same length, specifying the coordinates of
            the points in the scatter plot.
        labels:
            If not None, this has to be an array of same lenght as
            `xvals` and `yvals`.  Points in the scatter plot are then
            colored according to their label.
        """

        if data is not None:
            if data.shape[1] == 2:
                xvals, yvals = data[:, 0], data[:, 1]
            elif data.shape[1] == 3:
                xvals, yvals, labels = data[:, 0], data[:, 1], data[:, 2]
            else:
                raise ValueError("Bad data shape for 2D scatter plot: "
                                 f"{data.shape}")

        self._plot(xvals, yvals, labels)

    def _plot(self, xvals, yvals, labels) -> None:
        # FIXME[hack]: deal with synchronization problems: don't plot if
        # plotter has been deactivated ... find a better solution!
        figure = self._figure
        axes = self._axes
        if figure is None or axes is None:
            return

        if self._scatter_2d is None:
            figure.set_figwidth(8)
            figure.set_figheight(8)

            if labels is None:
                self._scatter_2d = \
                    axes.scatter(xvals, yvals, marker='.',
                                 edgecolor='none')
            else:
                number_of_labels = np.max(labels) + 1
                self._colormap = self._discrete_cmap('jet', number_of_labels)
                self._scatter_2d = \
                    axes.scatter(xvals, yvals, c=labels, marker='.',
                                 cmap=self._colormap, edgecolor='none',
                                 vmin=-.5, vmax=number_of_labels-.5)
                divider = make_axes_locatable(self._axes)
                self._cax = divider.append_axes('right', size='5%', pad=0.05)
                self._colorbar = \
                    self._figure.colorbar(self._scatter_2d, cax=self._cax,
                                          orientation='vertical')
                self._colorbar.ax.get_yaxis().set_ticks(range(number_of_labels))
                # cbar.ax.get_yaxis().set_ticklabels(range(string_labels))

        else:
            self._scatter_2d.set_offsets(np.c_[xvals, yvals])
            if labels is not None:
                number_of_labels = np.max(labels) + 1
                mapable = mpl.cm.ScalarMappable(cmap=self._colormap)  # norm=n, 
                self._scatter_2d.set_facecolor(mapable.to_rgba(labels))
            # recompute the ax.dataLim
            #axes.relim()
            # update ax.viewLim using the new dataLim
            #axes.autoscale_view()
            axes.set_xlim(xvals.min(), xvals.max())
            axes.set_ylim(yvals.min(), yvals.max())
            #plt.draw()

        # add some gridlines on the plot
        plt.grid()

    @staticmethod
    def _discrete_cmap(base_name: str = None, steps: int = 10):
        """Create a discrete color map.  The color map is created by
        taking colors in regular intervals from a given (continuous)
        base color map.
        """
        base_cmap = plt.cm.get_cmap(base_name)
        color_list = base_cmap(np.linspace(0, 1, steps))
        cmap_name = base_cmap.name + str(steps)
        return base_cmap.from_list(cmap_name, color_list, steps)

