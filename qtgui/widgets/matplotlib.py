import sys
import time

import numpy as np

#from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from matplotlib.figure import Figure

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout

class QMatplotlib(FigureCanvas):
    
    def __init__(self, parent=None, figsize=(8, 3)):
        super().__init__(Figure(figsize))

        # How to accept keyboard focus:
        #  Qt.NoFocus: accept no focus at all [default]
        #  Qt.TabFocus: focus by tabbing clicking
        #  Qt.ClickFocus: focus by mouse clicking
        #  Qt.StrongFocus = both (Qt.TabFocus or Qt.ClickFocus)
        self.setFocusPolicy(Qt.ClickFocus)

        # Actively grab the focus
        #self.setFocus()

        # Matplotlib events:
        # https://matplotlib.org/users/event_handling.html
        cid_key = self.mpl_connect('key_press_event', self._onKeyPress)
        cid_mouse = self.mpl_connect('button_press_event', self._onMousePress)

        # Figure.subplots is a new feature in Matplotlib 2.1.
        #self._ax = self.figure.subplots()
        self._ax = self.figure.add_subplot(111)

        #
        # Place some default content
        #
        self.showMessage('matplotlib')
        
        #t = np.linspace(0, 10, 501)
        #self._ax.plot(t, np.tan(t), ".")

        #layout = QVBoxLayout(self)
        #
        #static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        #layout.addWidget(static_canvas)
        #self.addToolBar(NavigationToolbar(static_canvas, self))
        #
        #dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        #layout.addWidget(dynamic_canvas)
        #self.addToolBar(QtCore.Qt.BottomToolBarArea,
        #                NavigationToolbar(dynamic_canvas, self))
        # ...
        #self._dynamic_ax = dynamic_canvas.figure.subplots()
        #self._timer = dynamic_canvas.new_timer(
        #    100, [(self._update_canvas, (), {})])
        #self._timer.start()

    def __enter__(self):
        self._ax.clear()
        return self._ax

    def __exit__(self, type, value, traceback):
        self._ax.figure.canvas.draw()

    def _onKeyPress(self, event):
        print(f"Matplotlib: you pressed '{event.key}'")

    def _onMousePress(self, event):
        click = 'double' if event.dblclick else 'single'
        button = event.button
        pixel_x = event.x
        pixel_y = event.y
        data_x = event.xdata
        data_y = event.ydata
        print(f"Matplotlib: {click} click with button {button}"
              f" x={pixel_x}, y={pixel_y}," +
              ("None" if data_x is None else f"data=({data_x},{data_y})"))

    def _update_canvas(self):
        # FIXME[old]
        self._ax.clear()
        t = np.linspace(0, 10, 101)
        # Shift the sinusoid as a function of time.
        self._ax.plot(t, np.sin(t + time.time()))
        self._ax.figure.canvas.draw()

    def scatter(self, *args, **kwargs):
        with self as ax:
            ax.scatter(*args, **kwargs)

    def imshow(self, *args, **kwargs):
        with self as ax:
            ax.imshow(*args, **kwargs)

    def plot(self, *args, **kwargs):
        with self as ax:
            ax.plot(*args, **kwargs)

    def showMessage(self, message: str):
        with self as ax:
            ax.axis('off')
            ax.text(0.5, 0.5, message,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)

    def noData(self):
        self.showMessage("No data")
