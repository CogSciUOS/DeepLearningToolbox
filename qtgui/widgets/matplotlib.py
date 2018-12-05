import sys
import time

import numpy as np

#from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from matplotlib.figure import Figure

from PyQt5.QtWidgets import QWidget, QVBoxLayout


class QMatplotlib(FigureCanvas):
    def __init__(self, parent=None, figsize=(8, 3)):
        super().__init__(Figure(figsize))
        #layout = QVBoxLayout(self)

        #static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        #layout.addWidget(static_canvas)
        #self.addToolBar(NavigationToolbar(static_canvas, self))

        #dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        #layout.addWidget(dynamic_canvas)
        #self.addToolBar(QtCore.Qt.BottomToolBarArea,
        #                NavigationToolbar(dynamic_canvas, self))

        self._ax = self.figure.subplots()
        t = np.linspace(0, 10, 501)
        self._ax.plot(t, np.tan(t), ".")

        #self._dynamic_ax = dynamic_canvas.figure.subplots()
        #self._timer = dynamic_canvas.new_timer(
        #    100, [(self._update_canvas, (), {})])
        #self._timer.start()

    def _update_canvas(self):
        self._ax.clear()
        t = np.linspace(0, 10, 101)
        # Shift the sinusoid as a function of time.
        self._ax.plot(t, np.sin(t + time.time()))
        self._ax.figure.canvas.draw()
