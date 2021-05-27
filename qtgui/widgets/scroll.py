# Qt imports
from PyQt5.QtCore import Qt, QObject, QEvent, QSize
from PyQt5.QtWidgets import QScrollArea, QSizePolicy


class QOrientedScrollArea(QScrollArea):
    """An auxiliary class derived from `QScrollArea` to handle cases,
    where scrolling is only supposed to happen into one direction.
    The content widget may change its size in both directions, and
    if it resizes into the direction without scrollbar, then this
    :py:class:`QOrientedScrollArea` should change its size accordingly,
    to adapt to that new size.
    """
    def __init__(self, orientation: Qt.Orientation = Qt.Horizontal,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._orientation = orientation
        if orientation == Qt.Horizontal:
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.setSizePolicy(QSizePolicy.MinimumExpanding,
                               QSizePolicy.Fixed)
        else:  # orientation == Qt.Vertical:
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.setSizePolicy(QSizePolicy.Fixed,
                               QSizePolicy.MinimumExpanding)

    def orientation(self) -> Qt.Orientation:
        """Get the orientation of this
        :py:class:`QOrientedScrollArea`.
        """
        return self._orientation

    def setOrientation(self, orientation: Qt.Orientation) -> None:
        """Set the orientation of this
        :py:class:`QOrientedScrollArea`.
        """
        if orientation != self._orientation:
            self._orientation = orientation
            self.update()

    def eventFilter(self, obj: QObject, event: QEvent):
        """Filter events in order to get informed when the content
        widget resizes.
        """
        if obj == self.widget() and event.type() == QEvent.Resize:
            self.updateGeometry()
        return super().eventFilter(obj, event)

    def sizeHint(self) -> QSize:
        """The size hint determines the exact size along the axis orthogonal
        to the orientation of this
        :py:class:`QOrientedScrollArea`.
        """
        widget = self.widget()
        if self.widget is None:
            return super().sizeHint()
        widgetSize = widget.size()

        margins = self.contentsMargins()
        marginsWidth = margins.left() + margins.right()
        marginsHeight = margins.top() + margins.bottom()

        if self._orientation == Qt.Horizontal:
            size = QSize(widgetSize.width() + marginsWidth,
                         widgetSize.height() + marginsHeight +
                         self.horizontalScrollBar().sizeHint().height())
        else:  # self._orientation == Qt.Vertical:
            size = QSize(widgetSize.width() + marginsWidth +
                         self.verticalScrollBar().sizeHint().width(),
                         widgetSize.height() + marginsHeight)
        return size
