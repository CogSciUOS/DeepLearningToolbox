"""
"""

# standard imports
from typing import Tuple
import time
import logging

# Qt imports
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QApplication,
                             QHBoxLayout, QRadioButton, QStyle, QScrollBar,
                             QCheckBox)
from PyQt5.QtGui import (QPainter, QPainterPath, QPen,
                         QMouseEvent, QWheelEvent, QResizeEvent, QPaintEvent)

# toolbox imports
from dltb.base.sound import (Sound, SoundPlayer, SoundRecorder,
                             SoundDisplay)
from dltb.util.time import time_str
from ..utils import QObserver, protect, QThreadedUpdate, pyqtThreadedUpdate

# logging
LOG = logging.getLogger(__name__)


class QSoundViewer(QThreadedUpdate, QObserver, SoundDisplay,
                   qobservables={
        SoundPlayer: {'state_changed', 'position_changed'}}):
    """A Qt-based graphical widget that allows to display sound.
    """
    MODE_PLAYING = 1
    MODE_RECORDING = 2

    MINIMUM_VIEW_LENGTH = 1.0  # one second

    viewChanged = pyqtSignal()

    def __init__(self, sound: Sound = None, player: SoundPlayer = None,
                 **kwargs) -> None:
        LOG.info("Initializing QSoundViewer(sound=%s, player=%s)",
                 sound, player)
        super().__init__(**kwargs)
        print(self.__class__.__mro__)
        self._sound = sound
        self._player = player
        self._mode = self.MODE_PLAYING
        self._position = None
        self._lastUpdateTime = 0.
        self._refresh = .1  # refresh rate in seconds (None = immedeate)
        self._path = None
        self._view = (0.0, 1.0 if sound is None else sound.duration)
        self._selection = None

    def length(self) -> float:
        return 0. if self._sound is None else self._sound.duration

    def position(self) -> float:
        return self._position

    def setPosition(self, position: float) -> None:
        if position != self._position:
            self._position = position
            self.update()

    def selection(self) -> Tuple[float, float]:
        return self._selection

    def setSelection(self, start: float, end: float) -> None:
        if (start, end) != self._selection:
            self._selection = (start, end)
            self.update()

    def view(self) -> Tuple[float, float]:
        return self._view

    def setView(self, start: float = None, end: float = None) -> None:
        """Set the view shown by this :py:class:`QSoundViewer`.

        Arguments
        ---------
        start: float
            The start time of the view (in seconds).
        end: float
            The end time of the view (in seconds).
        """
        if start is None or start < 0:
            start = 0
        if end is None or end > self._sound.duration:
            end = self._sound.duration

        if end-start < self.MINIMUM_VIEW_LENGTH:
            end = start + self.MINIMUM_VIEW_LENGTH

        if self._view != (start, end):
            self._view = (start, end)
            self._path = None
            self.update()
            self.viewChanged.emit()

    def mode(self) -> int:
        return self._mode

    def setMode(self, mode: int) -> None:
        self._mode = mode
        self.update()

    @protect
    def resizeEvent(self, event: QResizeEvent) -> None:
        """Process resize events. Resizing the widget requires
        an update of the wave displayed.

        Parameters
        ----------
        event: QResizeEvent
        """
        # FIXME[bug]: resizing the window during playback causes a crash:
        #
        #  src/hostapi/alsa/pa_linux_alsa.c:3636:
        #      PaAlsaStreamComponent_BeginPolling:
        #          Assertion `ret == self->nfds' failed.
        #
        # This is probably due to an empty buffer which may be caused by
        # the GUI repainting of the path takes too much time and slows
        # down the buffer filling.
        #
        # No: it actually seems that painting a path with too many points
        # is taking to much time, even if the path is precomputed. So
        # what we actually should do is limit the number of points
        # (and hence the level of detail) to be displayed, at least
        # during playback.
        # Other options:
        # - only repaint relevant parts of the wave curve (not an option
        #   on resize)
        # - increase the buffer size / number of buffers or priority
        #   for sound playback
        self.updatePath()

    @pyqtThreadedUpdate
    def updatePath(self) -> None:
        MAXIMAL_SAMPLE_POINTS = 200  # check when playback breaks
        width = min(self.width(), MAXIMAL_SAMPLE_POINTS)
        x_ratio = self.width()/width
        height = self.height()
        #text_height = self.fontMetrics().height()
        text_height = 20

        level = self._sound.level(width, start=self._view[0],
                                  end=self._view[1])
        level = (1-2*level) * (height - text_height)
        path = QPainterPath()
        iterator = enumerate(level)
        path.moveTo(*next(iterator))
        for (x,y) in iterator:
            path.lineTo(x*x_ratio, y)
        self._path = path
        self.update()

    @protect
    def paintEvent(self, event: QPaintEvent) -> None:
        """Process the paint event by repainting this Widget.

        Parameters
        ----------
        event: QPaintEvent
        """
        if self._sound is None:
            return

        # FIXME[bug?]: this methods seems to be invoked quite often
        # - check if this is so and why!
        painter = QPainter()
        painter.begin(self)

        #transform = QTransform()
        #transform.translate(x, y)
        #transform.scale(w_ratio, h_ratio)
        #painter.setTransform(transform)

        pen_width = 2  # 1
        pen_color = Qt.blue  # Qt.green
        pen = QPen(pen_color)
        pen.setWidth(pen_width)
        painter.setPen(pen)

        if self._mode == self.MODE_PLAYING:
            self._paintSoundLevel(painter)
        elif self._mode == self.MODE_RECORDING:
            self._paintSoundRecording(painter)

        # polygon = QPolygonF(map(lambda p: QPointF(*p), enumerate(wave)))
        # painter.drawPolyline(polygon)
        #
        painter.end()

    def _paintSoundLevel(self, painter: QPainter) -> None:
        """Paint wave (power)
        """
        fontMetrics = painter.fontMetrics()
        pen = painter.pen()
        width = self.width()
        height = self.height()
        text_height = fontMetrics.height()

        # write times
        pen.setColor(Qt.black)
        start_string = time_str(self._view[0])
        end_string = time_str(self._view[1])
        painter.drawText(0, height, start_string)
        painter.drawText(width - fontMetrics.width(end_string),
                         height, end_string)

        # draw sound wave
        if self._path is not None:
            painter.drawPath(self._path)

        # draw position indicator
        if self._player is not None:
            position = self._player.position
            if position is not None:
                x_position = int(((position - self._view[0]) /
                                  (self._view[1] - self._view[0])) * width)
                if 0 <= x_position <= width:
                    # draw vertical line
                    painter.setPen(QPen(Qt.red, 1))
                    painter.drawLine(x_position, 0, x_position, height)
                    # write time
                    position_string = time_str(self._player.position)
                    text_width = fontMetrics.width(position_string)
                    x_location = max(0, min(x_position - text_width // 2,
                                            width - text_width))
                    painter.drawText(x_location, text_height, position_string)

    def _paintSoundRecording(self, painter: QPainter,
                             downsample: int = 10) -> None:
        points = self.width()
        samplerate = self._sound.samplerate / downsample

        duration = points / samplerate

        if self._position is None:
            start = max(0, self._sound.duration - duration)
        else:
            start = self._position
        end = min(start + duration, self._sound.duration)

        # get the sound wave
        wave = self._sound[start:end:samplerate]

        if len(wave) > 0:
            wave = (wave[:, 0] + 1.0) * (0.5*self.height())
            path = QPainterPath()
            path.moveTo(0, wave[0])
            for p in enumerate(wave):
                path.lineTo(*p)
            painter.drawPath(path)

    def set_sound(self, sound: Sound) -> None:
        LOG.info("Setting Sound for QSoundViewer: %s", sound)
        self._sound = sound
        self.update()

    @protect
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """A mouse press toggles between raw and processed mode.
        """
        if self._player is not None:
            position = (self._view[0] + (event.x() / self.width()) *
                        (self._view[1]-self._view[0]))
            self._player.position = position
            self.update()

    def FIXME_demo_animation_loop(self):
        """This function does not anything useful - it is just meant as a
        demonstration how an animation loop in PyQt5 could be realized.

        """
        while True:  # need some stop criterion ...
            # do something (update data)
            self.update()  # initiate update of display (i.e., repaint)
            QApplication.processEvents()  # start the actual repainting
            time.sleep(0.0025)  # wait a bit

    def wheelEvent(self, event: QWheelEvent):
        """Process mouse wheel events. The mouse wheel can be used for
        zooming.

        Parameters
        ----------
        event: QWheelEvent
            The event providing the angle delta.
        """

        delta = event.angleDelta().y() / 120  # will be +/- 1

        center = (self._view[0] + (event.x() / self.width() *
                                     (self._view[1] - self._view[0])))
        end = center - (center - self._view[1]) / (1 + delta * 0.01)
        start = center - (center - self._view[0]) / (1 + delta * 0.01)

        self.setView(start, end)

        # We will accept the event, to prevent interference
        # with the QScrollArea.
        event.accept()

    #
    # Plotter interface
    #

    def start_plot(self) -> None:
        pass

    def stop_plot(self) -> None:
        pass

    def player_changed(self, player: SoundPlayer,
                       info: SoundPlayer.Change) -> None:
        if info.position_changed:
            LOG.debug("QSoundViewer: player position changed: %s",
                      player.position)
            self._position = player.position
            currentTime = time.time()
            if (self._refresh is None or
                currentTime - self._lastUpdateTime > self._refresh):
                # update if enough time has passed since last refresh
                self._lastUpdateTime = currentTime
                self.update()



class QSoundViewerScrollbar(QScrollBar):

    def __init__(self, soundViewer: QSoundViewer, **kwargs) -> None:
        super().__init__(Qt.Horizontal, **kwargs)
        self._soundViewer = None
        self.setSoundViewer(soundViewer)
        self.valueChanged.connect(self.onValueChanged)

    def setSoundViewer(self, soundViewer: QSoundViewer) -> None:
        if self._soundViewer == soundViewer:
            return  # nothing changed

        if self._soundViewer is not None:
            self._soundViewer.viewChanged.disconnect(self.onViewChanged)

        self._soundViewer = soundViewer

        if self._soundViewer is not None:
            self._adaptSlider()
            self._soundViewer.viewChanged.connect(self.onViewChanged)

    def _adaptSlider(self) -> None:
        length = self._soundViewer.length()
        view = self._soundViewer.view()
        view_length = view[1] - view[0]
        maximum = length-view_length
        self.setMaximum(int(1000 * maximum))
        self.setPageStep(int(1000 * view_length))
        self.setSliderPosition(int(1000 * view[0]))

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.drawText(self.width()-100, 10, "Hallo")
        painter.end()
        super().paintEvent(event)
        painter = QPainter()
        painter.begin(self)
        painter.drawText(20, 10, "Hallo")
        painter.end()

    @pyqtSlot()
    def onViewChanged(self) -> None:
        self._adaptSlider()

    @pyqtSlot(int)
    def onValueChanged(self, value: int) -> None:
        position = value / 1000
        length = self.pageStep() / 1000
        self._soundViewer.setView(position, position+length)


class QSoundControl(QWidget, QObserver, qobservables={
        SoundPlayer: {'state_changed'}}):
    """A Qt-based graphical widget that allows to control playback
    and recording of sounds.
    """

    def __init__(self, sound: Sound,
                 player: SoundPlayer = None,
                 recorder: SoundRecorder = None,
                 **kwargs) -> None:
        LOG.info("Initializing QSoundControl(sound=%s, player=%s, "
                 "recorder=%s)", sound, player, recorder)
        super().__init__(**kwargs)

        self._sound = sound

        self._player = player
        self.observe(player)
        
        self._recorder = recorder
        self.observe(recorder)

        style = self.style()
        self._iconPlay = style.standardIcon(getattr(QStyle, 'SP_MediaPlay'))
        self._iconPause = style.standardIcon(getattr(QStyle, 'SP_MediaPause'))

        layout = QVBoxLayout()
        self._buttonRecord = QPushButton("Record")
        self._buttonRecord.setCheckable(True)
        self._buttonRecord.clicked.connect(self._onButtonRecordClicked)
        layout.addWidget(self._buttonRecord)

        self._buttonPlay = QPushButton("Play")
        self._buttonPlay.setCheckable(True)
        self._buttonPlay.clicked.connect(self._onButtonPlayClicked)
        self._buttonPlay.setIcon(self._iconPlay)
        layout.addWidget(self._buttonPlay)

        self._buttonInfo = QPushButton("Info")
        self._buttonInfo.clicked.connect(self._onButtonInfoClicked)
        layout.addWidget(self._buttonInfo)

        self._soundViewer = QSoundViewer(self._sound, self._player)
        self._soundViewer.setMinimumSize(200, 200)
        self._soundViewer.observe(self._player)
        layout.addWidget(self._soundViewer)
        layout.addWidget(QSoundViewerScrollbar(self._soundViewer))

        # self._matplotlib = QMatplotlib()
        # layout.addWidget(self._matplotlib)
        self._plotter = self._soundViewer
        # self._plotter = None
        #self._plotter = MatplotlibSoundPlotter(samplerate=sound.samplerate,
        #                                       standalone=False,
        #                                       figure=self._matplotlib.figure,
        #                                       ax=self._matplotlib._ax)

        radioLayout = QHBoxLayout()
        self.b1 = QRadioButton("Playing")
        self.b1.setChecked(True)
        self.b1.toggled.connect(lambda: self.btnstate(self.b1))
        radioLayout.addWidget(self.b1)

        self.b2 = QRadioButton("Recording")
        self.b2.toggled.connect(lambda: self.btnstate(self.b2))
        radioLayout.addWidget(self.b2)

        self._checkboxLoop = QCheckBox("Loop")
        self._checkboxLoop.stateChanged.connect(self._onLoopChanged)
        radioLayout.addWidget(self._checkboxLoop)

        self._checkboxReverse = QCheckBox("Reverse")
        self._checkboxReverse.stateChanged.connect(self._onReverseChanged)
        radioLayout.addWidget(self._checkboxReverse)

        layout.addLayout(radioLayout)

        self.setLayout(layout)

    def btnstate(self, b):
        if b == self.b1:
            self._soundViewer.setMode(self._soundViewer.MODE_PLAYING)
        if b == self.b2:
            self._soundViewer.setMode(self._soundViewer.MODE_RECORDING)

    @pyqtSlot(bool)
    # @protect
    def _onButtonRecordClicked(self, checked: bool) -> None:
        if self._recorder is None:
            print("QSoundControl: No recorder, sorry!")
        elif checked:
            print("QSoundControl: Recording sound")
            recorder.record(self._sound)
        else:
            print("QSoundControl: Stop recording sound")
            self._recorder.stop()

    @pyqtSlot(bool)
    # @protect
    def _onButtonPlayClicked(self, checked: bool) -> None:
        if self._player is None:
            LOG.warning("QSoundControl: No player, sorry!")
        elif checked:
            LOG.info("QSoundControl: Playing sound %s on player %s",
                     self._sound, self._player)
            self._player.play(self._sound)
        else:
            LOG.info("QSoundControl: Stop playing sound on player %s",
                     self._player)
            self._player.stop()

    @pyqtSlot(bool)
    # @protect
    def _onButtonInfoClicked(self, checked: bool) -> None:
        print(f"info[QSoundControl]: Sound: {self._sound}")

    @pyqtSlot(int)
    # @protect
    def _onLoopChanged(self, state: int) -> None:
        if self._player is not None:
            self._player.loop = (state == Qt.Checked)

    @pyqtSlot(int)
    # @protect
    def _onReverseChanged(self, state: int) -> None:
        if self._player is not None:
            self._player.reverse = (state == Qt.Checked)

    def player_changed(self, player: SoundPlayer,
                       info: SoundPlayer.Change) -> None:
        if info.state_changed:
            self.update()
        
    def recorder_changed(self, player: SoundRecorder,
                         info: SoundRecorder.Change) -> None:
        if info.state_changed:
            self.update()

    def update(self) -> None:
        print(self._player is not None, self._player.sound)
        self._buttonPlay.setEnabled(self._player is not None and
                                    self._player.sound is not None)
        self._buttonPlay.setChecked(self._player is not None and
                                    self._player.playing)
        self._buttonRecord.setChecked(self._recorder is not None and
                                      self._recorder.recording)

        self._checkboxLoop.setEnabled(self._player is not None)
        self._checkboxLoop.setCheckState(Qt.Checked if
                                         self._player is not None and
                                         self._player.loop else Qt.Unchecked)
        self._checkboxReverse.setEnabled(self._player is not None)
        self._checkboxReverse.setCheckState(Qt.Checked if
                                            self._player is not None and
                                            self._player.reverse else
                                            Qt.Unchecked)
        super().update()
