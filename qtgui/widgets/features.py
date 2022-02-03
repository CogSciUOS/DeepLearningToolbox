"""Widgets for working with feature vectors.
"""

# third-party imports
import numpy as np

# Qt imports
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtCore import QSize, QPoint, QRect
from PyQt5.QtGui import QMouseEvent, QKeyEvent, QFocusEvent, QPaintEvent
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QWidget, QFrame, QLabel
from PyQt5.QtWidgets import QGridLayout, QSizePolicy, QToolTip

# GUI imports
from ..utils import protect


class QFeatureView(QFrame):
    """A :py:class:`QFeatureView` graphically displays a feature vectors
    and provides basic editing capabilities.
    """

    # FIXME[todo]: zoom does not work yet

    Clear: int = 0
    Random: int = 1
    Increase: int = 2
    Decrease: int = 3
    Invert: int = 5
    Positive: int = 5
    Negative: int = 6

    featuresChanged = pyqtSignal()

    def __init__(self, orientation: int = Qt.Horizontal, **kwargs) -> None:
        """Initialize the :py:class:`QFeatureView`.
        """
        super().__init__(**kwargs)

        self._range = (0., 0.)
        self._features = None

        #
        # edit
        #
        self._edit = False
        # editPallete = QPalette()
        # editPallete.setColor(QPalette.Background, Qt.blue.lighter)
        # self._editPalletes = (self.palette(), editPallete)

        #
        # zoom
        #
        self._zoom = False
        self._zoomIndex = None
        self._zoomWidth = 10

        # layout
        self._orientation = orientation
        if orientation == Qt.Horizontal:
            self.setSizePolicy(QSizePolicy.MinimumExpanding,
                               QSizePolicy.Preferred)
        else:
            self.setSizePolicy(QSizePolicy.Preferred,
                               QSizePolicy.MinimumExpanding)
        self.setStatusTip("Feature values.")
        self.setWhatsThis("This widget displays the feature values.")
        self.featuresChanged.connect(self.updateFeatures)

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by 'Tab' key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)

    def setFeatures(self, features: np.ndarray) -> None:
        self._features = features
        max_value = 0 if features is None else np.abs(self._features).max()
        self._range = (-max_value, max_value)
        self.update()

    def minimumSizeHint(self) -> QSize:
        """The minimum size hint.

        Returns
        -------
        QSize : The minimal size of this :py:class:`QFeatureView`
        """
        features = 200 if self._features is None else len(self._features)
        return QSize(features if self._orientation == Qt.Horizontal else 200,
                     200 if self._orientation == Qt.Horizontal else features)

    def indexAtPosition(self, position: QPoint) -> float:
        offset = (position.x() if self._orientation == Qt.Horizontal else
                  position.y())
        total = (self.width() if self._orientation == Qt.Horizontal else
                 self.height())
        zoom = self._zoomIndex
        displayIndex = offset * self._displayFeatures(zoom) // total
        return self._realIndex(displayIndex, zoom)

    def _displayFeatures(self, zoom: int) -> int:
        if self._features is None:
            return 1
        number = len(self._features)
        if zoom is not None:
            if zoom < self._zoomWidth:
                delta = self._zoomWidth - zoom
            elif zoom + self._zoomWidth > number:
                delta = zoom + self._zoomWidth - number
            else:
                delta = 0
            number += self._zoomWidth * (self._zoomWidth - 1)
            number -= delta * (delta - 1)
        return number

    def _realIndex(self, index: int, zoom: int) -> int:
        if zoom is None:
            return index
        triangle = self._zoomWidth * (self._zoomWidth + 1) // 2
        b1 = zoom - self._zoomWidth
        b2 = b1 + triangle
        b3 = b2 + triangle
        if index < b1:
            return index
        from math import sqrt
        if index < b2:
            delta = b2 - index
            return b1 + int(1 + sqrt(1+8*delta)/2)
        if index < b3:
            delta = index - b2
            return zoom + self._zoomWidth - int(1 + sqrt(1+8*delta)/2)
        return zoom + self._zoomWidth + (index - b3)

    def valueAtPosition(self, position: QPoint) -> float:
        offset = (self.height() - position.y()
                  if self._orientation == Qt.Horizontal else position.x())
        total = (self.height()
                 if self._orientation == Qt.Horizontal else self.width())
        value = (self._range[0] +
                 (offset * (self._range[1]-self._range[0]) / total))
        return value

    def paintEvent(self, event: QPaintEvent) -> None:
        """Process the paint event by repainting this Widget.

        Parameters
        ----------
        event : QPaintEvent
        """
        super().paintEvent(event)  # make sure the frame is painted

        if self._features is not None:
            painter = QPainter()
            painter.begin(self)
            self._drawFeatures(painter, event.rect())
            painter.end()

    def _drawFeatures(self, painter: QPainter, rect: QRect) -> None:
        """Draw a given portion of this widget.
        Parameters
        ----------
        painter : QPainter
        rect : QRect
        """
        # mind the frame ...
        padding = self.lineWidth() + 1
        x1 = padding
        y1 = padding
        x2 = self.width() - padding
        y2 = self.height() - padding
        width = x2 - x1
        height = y2 - y1

        pen = QPen(Qt.red if self.edit() else Qt.black)
        pen_width = 1
        pen.setWidth(pen_width)
        
        if self._orientation == Qt.Horizontal:
            painter.drawText(x1, y1+10, f"{self._range[1]:.3f}")
            painter.drawText(x1, y2, f"{self._range[0]:.3f}")
            center_y = height // 2
            stretch_y = center_y / self._range[1]

            if self._zoomIndex is None:
                stretch_x = width / len(self._features)
                pen_width = max(1, stretch_x-1)
                pen.setWidth(pen_width)
                if pen_width > 2:
                    pen.setColor(Qt.gray)
                painter.setPen(pen)
                for x, feature in enumerate(self._features):
                    pos_x = int(x*stretch_x)
                    pos_y = int(center_y - feature * stretch_y)
                    painter.drawLine(pos_x, center_y, pos_x, pos_y)
            else:
                zoom = self._zoomIndex
                stretch_x = width / self._displayFeatures(zoom)
                index1 = zoom - self._zoomWidth
                index2 = zoom + self._zoomWidth
                pos_x = 0.
                for index, feature in enumerate(self._features):
                    feature_width = stretch_x
                    if index1 <= index < index2:
                        extra_width = self._zoomWidth - abs(zoom - index)
                        feature_width *= extra_width
                    pen.setWidth(max(feature_width, pen_width))
                    painter.setPen(pen)
                    pos_y = center_y - feature * stretch_y
                    painter.drawLine(int(pos_x), center_y,
                                     int(pos_x), int(pos_y))
                    pos_x += feature_width
        else:
            width = self.width()
            center_x = width // 2
            stretch_x = center_x / self._range[1]

            stretch_y = height / len(self._features)
            for y, feature in enumerate(self._features):
                pos_x = int(center_x + feature * stretch_x)
                pos_y = int(y*stretch_y)
                painter.drawLine(center_x, pos_y, pos_x, pos_y)

    @protect
    def mouseMoveEvent(self, event: QMouseEvent):
        """Process mouse movements.  If tooltips are active, information on
        the entry at the current mouse position are displayed.

        Attention: The mouseMoveEvent() is only called for regular
        mouse movements, if mouse tracking is explicitly enabled for
        this widget by calling self.setMouseTracking(True).  Otherwise
        it may be called on dragging.

        Parameters
        ----------
        event : QMouseEvent
            The mouse event, providing local and global coordinates.
        """
        position = event.pos()
        index = self.indexAtPosition(position)
        currentValue = 0 if self._features is None else self._features[index]
        mouseValue = self.valueAtPosition(position)
        QToolTip.showText(event.globalPos(), f"{event.pos()}: "
                          f"features[{index}] = {currentValue:.3f} "
                          f"[->{mouseValue:.3f}]")

        if self._zoom:
            cursor = self.cursor()
            position = cursor.pos()
            print(f"Cursor.pos: {cursor.pos()}")
            cursor.setPos(position.x()+10, position.y()+10)
            self._zoomIndex = index
            self.update()

    @protect
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._features is None:
            return  # nothing to do

        if self._edit:
            position = event.pos()
            index = self.indexAtPosition(position)
            self._features[index] = self.valueAtPosition(position)
            self.featuresChanged.emit()

    @protect
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.setMouseTracking(not self.hasMouseTracking())

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`QImageView` supports
        the following keys:

        c/0: clear features
        r: random features
        +: increase features
        -: decrease features
        i: invert features
        p: positive features
        n: negative features
        z: toggle zoom mode
        e: toggle edit mode
        """
        key = event.key()

        if key in (Qt.Key_C, Qt.Key_0):  # Clear
            self.adaptFeatures(self.Clear)
        elif key == Qt.Key_R:  # Random
            self.adaptFeatures(self.Random)
        elif key == Qt.Key_Plus:  # Increase
            self.adaptFeatures(self.Increase)
        elif key == Qt.Key_Minus:  # Decrease
            self.adaptFeatures(self.Decrease)
        elif key == Qt.Key_I:  # Invert
            self.adaptFeatures(self.Invert)
        elif key == Qt.Key_P:  # Positive
            self.adaptFeatures(self.Positive)
        elif key == Qt.Key_N:  # Negative
            self.adaptFeatures(self.Negative)
        elif key == Qt.Key_Z:  # zoom
            self.setZoom(not self.zoom())
        elif key == Qt.Key_E:  # edit
            self.setEdit(not self.edit())

    def adaptFeatures(self, adaptation: int) -> None:
        features = self._features
        if features is None:
            return  # nothing to adapt ...
        if adaptation == self.Clear:  # x -> 0
            features[:] = 0
        elif adaptation == self.Random:  # x -> random normal
            features[:] = np.random.randn(*features.shape)
        elif adaptation == self.Increase:  # x -> x * 1.1
            features[:] = features[:] * 1.1
        elif adaptation == self.Decrease:  # x-> x * 0.9
            features[:] = features[:] * 0.9
        elif adaptation == self.Invert:  # invert: x -> -x
            features[:] = -features[:]
        elif adaptation == self.Positive:  # positive: x -> max(0,x)
            features[features < 0] = 0
        elif adaptation == self.Negative:  # negative: x -> min(0,x)
            features[features > 0] = 0
        else:
            raise ValueError(f"Unknow adaptation: {adaptation}")
        self.featuresChanged.emit()

    def focusInEvent(self, event: QFocusEvent) -> None:
        self.setFrameStyle(QFrame.Box)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        self.setFrameStyle(QFrame.NoFrame)

    def edit(self) -> bool:
        return self._edit

    def setEdit(self, edit: bool) -> None:
        self._edit = edit
        self.update()

    def zoom(self) -> bool:
        return self._zoom

    def setZoom(self, zoom: bool) -> None:
        self._zoom = zoom
        self._zoomIndex = None
        self.setMouseTracking(zoom)
        self.update()

    @pyqtSlot()
    def updateFeatures(self) -> None:
        self.update()


class QFeatureInfo(QWidget):
    """The :py:class:`QFeatureInfo` textually displays statistics of a
    feature vector.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the :py:class:`QFeatureInfo`.
        """
        super().__init__(**kwargs)
        self._initUI()
        self.setFeatures(None)

    def _initUI(self) -> None:
        grid = QGridLayout()
        # Dimensionality
        grid.addWidget(QLabel("Dimensionality"), 0, 0)
        self._dimensionality = QLabel()
        grid.addWidget(self._dimensionality, 0, 1)
        # Min/max
        grid.addWidget(QLabel("min/max"), 1, 0)
        self._minmax = QLabel()
        grid.addWidget(self._minmax, 1, 1)
        # Min/max
        grid.addWidget(QLabel("L2-norm"), 2, 0)
        self._l2norm = QLabel()
        grid.addWidget(self._l2norm, 2, 1)
        # density
        grid.addWidget(QLabel("density"), 3, 0)
        self._density = QLabel()
        grid.addWidget(self._density, 3, 1)

        # add the layout
        self.setLayout(grid)

    def setFeatures(self, features: np.ndarray) -> None:
        self._features = features
        self.update()

    def update(self) -> None:
        if self._features is None:
            self._dimensionality.setText('')
            self._minmax.setText('')
            self._l2norm.setText('')
            self._density.setText('')
        else:
            self._dimensionality.setText(f"{len(self._features)}")
            self._minmax.setText(f"{self._features.min():.4f}/"
                                 f"{self._features.max():.4f}")
            l2norm = np.linalg.norm(self._features)
            self._l2norm.setText(f"{l2norm:.2f}")
            # Gaussian density
            density = 1/np.sqrt(2*np.pi) * np.exp(-0.5 * l2norm**2)
            self._density.setText(f"{density:1.2e}")
        super().update()

    @pyqtSlot()
    def updateFeatures(self) -> None:
        self.update()
