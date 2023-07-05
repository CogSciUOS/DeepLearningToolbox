"""


:py:class:`QSoundViewer`

:py:class:`QSoundViewerScrollbar`

The :py:class:`QSoundControl` offers controls buttons
for a :py:class:`SoundPlayer` and/or :py:class:`SoundRecorder`


>>> from qtgui.widgets.sound import SoundDisplay
>>> display = SoundDisplay(sound='examples/win_xp_shutdown.wav', player=True)
>>> display.show()



"""

# standard imports
from typing import Tuple, Optional
import os
import time
import logging

# Qt imports
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QApplication,
                             QHBoxLayout, QRadioButton, QStyle, QScrollBar,
                             QCheckBox, QFileDialog, QGroupBox)
from PyQt5.QtGui import (QPainter, QPainterPath, QPen,
                         QMouseEvent, QWheelEvent, QResizeEvent, QPaintEvent)

# toolbox imports
from dltb.base.sound import Sound, Soundlike, SoundPlayer, SoundRecorder
from dltb.base.sound import SoundView, SoundDisplay
from dltb.base.gui import View
from dltb.util.time import time_str
from ..utils import QObserver, protect, QThreadedUpdate, pyqtThreadedUpdate
from .. import QStandalone

# logging
LOG = logging.getLogger(__name__)


class QSoundViewer(QThreadedUpdate, QObserver,
                   qobservables={
                       Sound: {'data_changed'},
                       SoundPlayer: {'state_changed', 'position_changed'}}):
    """A Qt-based graphical widget that allows to display sound.

    The sound will be displayed as a (subsampled) waveform.
    The `QSoundViewer` offers the possibility to zoom into the
    waveform.  The current view is provided by the :py:prop:`_view`
    property as a pair (first_position, last_position). It will
    initially be set to show the full sound, and it can be changed
    by :py:meth:`setView`.  The minimal view duration is fixed
    by the property `MINIMUM_VIEW_LENGTH`.
    

    The `QSoundViewer` can observe a :py:class:`SoundPlayer` to
    indicate the current playback position.


    The `QSoundViewer` can be in one of two modes: `MODE_PLAYING`
    will visualize playback, while `MODE_RECORDING` aims at the
    recording situation, with a changing :py:class:`Sound` object.
    """
    MODE_PLAYING = 1
    MODE_RECORDING = 2

    MINIMUM_VIEW_LENGTH = 1.0  # one second

    viewChanged = pyqtSignal()

    def __init__(self, sound: Optional[Sound] = None,
                 player: Optional[SoundPlayer] = None,
                 **kwargs) -> None:
        LOG.info("Initializing QSoundViewer(sound=%s, player=%s)",
                 sound, player)
        super().__init__(**kwargs)
        self.setSound(sound)
        self.setSoundPlayer(player)

        self._mode = self.MODE_PLAYING
        self._position = None
        self._lastUpdateTime = 0.
        self._refresh = .1  # refresh rate in seconds (None = immedeate)
        self._path = None
        self._view = (0.0, 1.0 if sound is None else sound.duration)
        self._selection = None

    def setSound(self, sound: Optional[Sound]) -> None:
        """Update the :py:class:`Sound` object displayed by this
        `QSoundViewer`.
        """
        LOG.info("Setting Sound for QSoundViewer: %s", sound)
        self._view = (0.0, 1.0 if sound is None else sound.duration)

    def setSoundPlayer(self, player: Optional[SoundPlayer]) -> None:
        """Update the :py:class:`SoundPlayer` object observed by this
        `QSoundViewer`.
        """
        LOG.info("Setting Player for QSoundViewer: %s", player)

    def length(self) -> float:
        """The length (duration) of the :py:class:`Sound` object
        displayed by this `QSoundViewer`. Will be 0.0 if no
        `Sound` object is displayed.
        """
        sound = self.sound()
        return 0. if sound is None else sound.duration

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
            The end time of the view (in seconds). If the end results
            in a view length less than `MINIMUM_VIEW_LENGTH`, it will
            be increased.
        """
        if start is None or start < 0:
            start = 0
        length = self.length()
        if end is None or end > length:
            end = length

        if end-start < self.MINIMUM_VIEW_LENGTH:
            end = start + self.MINIMUM_VIEW_LENGTH

        if self._view != (start, end):
            LOG.debug("setView(%.4f, %.4f)", start, end)
            self._view = (start, end)
            self._path = None
            self.updatePath()
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
        """Update the private `_path` property, holding a `QPainterPath`
        object used to display the sound wave.
        """

        sound = self.sound()

        if sound is not None:
            MAXIMAL_SAMPLE_POINTS = 200  # check when playback breaks
            width = min(self.width(), MAXIMAL_SAMPLE_POINTS)
            x_ratio = self.width()/width
            height = self.height()
            #text_height = self.fontMetrics().height()
            text_height = 20

            level = sound.level(width, start=self._view[0],
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
        sound = self.sound()
        if sound is None:
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
        player = self.soundPlayer()
        if player is not None:
            position = player.position
            if position is not None:
                x_position = int(((position - self._view[0]) /
                                  (self._view[1] - self._view[0])) * width)
                if 0 <= x_position <= width:
                    # draw vertical line
                    painter.setPen(QPen(Qt.red, 1))
                    painter.drawLine(x_position, 0, x_position, height)
                    # write time
                    position_string = time_str(player.position)
                    text_width = fontMetrics.width(position_string)
                    x_location = max(0, min(x_position - text_width // 2,
                                            width - text_width))
                    painter.drawText(x_location, text_height, position_string)

    def _paintSoundRecording(self, painter: QPainter,
                             downsample: int = 10) -> None:
        sound = self.sound()

        points = self.width()
        samplerate = sound.samplerate / downsample

        duration = points / samplerate

        if self._position is None:
            start = max(0, sound.duration - duration)
        else:
            start = self._position
        end = min(start + duration, sound.duration)

        # get the sound wave
        wave = sound[start:end:samplerate]

        if len(wave) > 0:
            wave = (wave[:, 0] + 1.0) * (0.5*self.height())
            path = QPainterPath()
            path.moveTo(0, wave[0])
            for p in enumerate(wave):
                path.lineTo(*p)
            painter.drawPath(path)

    @protect
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """A mouse press allows to set the players position.
        """
        player = self.soundPlayer()
        if player is not None:
            position = (self._view[0] + (event.x() / self.width()) *
                        (self._view[1]-self._view[0]))
            player.position = position
            self.update()

    @protect
    def wheelEvent(self, event: QWheelEvent) -> None:
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

    def data_changed(self, sound: Sound, info: Sound.Change) -> None:
        LOG.debug("QSoundViewer: sound changed: duration=%s",
                  sound.duration)

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

    def FIXME_demo_animation_loop(self):
        """This function does not anything useful - it is just meant as a
        demonstration how an animation loop in PyQt5 could be realized.

        """
        while True:  # need some stop criterion ...
            # do something (update data)
            self.update()  # initiate update of display (i.e., repaint)
            QApplication.processEvents()  # start the actual repainting
            time.sleep(0.0025)  # wait a bit


class QSoundViewerScrollbar(QScrollBar):
    """A special scroll bar for displaying a :py:class:`Sound` object.
    """

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
        if length == 0:
            position, maximum = 0, 0
            view_length = 0.001
        else:
            view = self._soundViewer.view()
            view_length = view[1] - view[0]
            maximum = length-view_length
            position = view[0]
        self.setMaximum(int(1000 * maximum))
        self.setPageStep(int(1000 * view_length))
        self.setSingleStep(int(100 * view_length))
        self.setSliderPosition(int(1000 * position))

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.drawText(self.width()-100, 10, "Hallo 2")
        painter.end()

        super().paintEvent(event)

        painter = QPainter()
        painter.begin(self)
        height = self.height()
        painter.drawText(20, height-3, "Hallo")

        if self._soundViewer is not None:
            length = self._soundViewer.length()
            position = self._soundViewer.position()
            if length is not None and position is not None:
                xPosition = self.width() * position/length
                painter.setPen(QPen(Qt.red, 1))
                painter.drawLine(xPosition, 0, xPosition, height)

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
        Sound: {'data_changed'},
        SoundPlayer: {'state_changed', 'position_changed'},
        SoundRecorder: {'state_changed'}}):
    """A Qt-based graphical widget that allows to control playback
    and recording of sounds.

    The `QSoundControl` is a graphical container bundling several
    components:

    * a `QSoundViewer`, combined with a `QSoundViewerScrollbar`, for
      displaying the `Sound` object.
    * a collection of buttons for controlling playback and recording.

    Properties
    ----------
    sound:
        The :py:class:`Sound` to be displayed in the sound viewer.
    player:
        A :py:class:`SoundPlayer` that can be used to play sounds.
        Controls for sound playback will only be shown/enabled,
        if a player is available.
    recorder:
        A :py:class:`SoundRecorder` that can be used to record sounds.
        Controls for sound recording will only be shown/enabled,
        if a player is available.
    """

    def __init__(self, sound: Optional[Sound] = None,
                 player: Optional[SoundPlayer] = None,
                 recorder: Optional[SoundRecorder] = None,
                 **kwargs) -> None:
        LOG.info("Initializing QSoundControl(sound=%s, player=%s, "
                 "recorder=%s)", sound, player, recorder)
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.update()

        self.setSoundPlayer(player)
        self.setSoundRecorder(recorder)
        self.setSound(sound)

    def _initUI(self) -> None:
        style = self.style()
        iconPlay = style.standardIcon(getattr(QStyle, 'SP_MediaPlay'))
        iconPause = style.standardIcon(getattr(QStyle, 'SP_MediaPause'))
        iconLoad = style.standardIcon(getattr(QStyle, 'SP_FileDialogStart'))

        self._buttonLoad = QPushButton("Load")
        self._buttonLoad.clicked.connect(self._onButtonLoadClicked)
        self._buttonLoad.setIcon(iconLoad)

        self._buttonRecord = QPushButton("Record")
        self._buttonRecord.setCheckable(True)
        self._buttonRecord.clicked.connect(self._onButtonRecordClicked)

        self._buttonPlay = QPushButton("Play")
        self._buttonPlay.setCheckable(True)
        self._buttonPlay.clicked.connect(self._onButtonPlayClicked)
        self._buttonPlay.setIcon(iconPlay)

        self._buttonInfo = QPushButton("Info")
        self._buttonInfo.clicked.connect(self._onButtonInfoClicked)

        self._soundViewer = QSoundViewer()
        self._soundViewer.setMinimumSize(200, 200)
        self.addAttributePropagation(Sound, self._soundViewer)
        self.addAttributePropagation(SoundPlayer, self._soundViewer)

        self._soundScrollbar = QSoundViewerScrollbar(self._soundViewer)
        
        # self._matplotlib = QMatplotlib()
        # layout.addWidget(self._matplotlib)
        self._plotter = self._soundViewer
        # self._plotter = None
        #self._plotter = MatplotlibSoundPlotter(samplerate=sound.samplerate,
        #                                       standalone=False,
        #                                       figure=self._matplotlib.figure,
        #                                       ax=self._matplotlib._ax)

    def _layoutUI(self) -> None:
        layout = QVBoxLayout()

        controls = QHBoxLayout()        
        controls.addWidget(self._buttonRecord)
        controls.addWidget(self._buttonLoad)
        controls.addWidget(self._buttonPlay)
        controls.addWidget(self._buttonInfo)
        layout.addLayout(controls)

        layout.addWidget(self._soundViewer)
        layout.addWidget(self._soundScrollbar)

        radioLayout = QHBoxLayout()
        self._b1 = QRadioButton("Playing")
        self._b1.setChecked(True)
        self._b1.toggled.connect(lambda: self.btnstate(self._b1))
        radioLayout.addWidget(self._b1)

        self._b2 = QRadioButton("Recording")
        self._b2.toggled.connect(lambda: self.btnstate(self._b2))
        radioLayout.addWidget(self._b2)

        self._checkboxLoop = QCheckBox("Loop")
        self._checkboxLoop.stateChanged.connect(self._onLoopChanged)
        radioLayout.addWidget(self._checkboxLoop)

        self._checkboxReverse = QCheckBox("Reverse")
        self._checkboxReverse.stateChanged.connect(self._onReverseChanged)
        radioLayout.addWidget(self._checkboxReverse)

        layout.addLayout(radioLayout)

        self.setLayout(layout)

    def btnstate(self, b):
        if b == self._b1:
            self._soundViewer.setMode(self._soundViewer.MODE_PLAYING)
        if b == self._b2:
            self._soundViewer.setMode(self._soundViewer.MODE_RECORDING)

    def setSound(self, sound: Optional[Sound]) -> None:
        player = self.soundPlayer()
        recorder = self.soundRecorder()
        if player is not None:
            player.sound = sound
        if recorder is not None:
            recorder.sound = sound
        self.update()

    def setSoundPlayer(self, player: Optional[SoundPlayer]) -> None:
        if player is not None:
            player.sound = self.sound()
        self.update()

    def setSoundRecorder(self, recorder: Optional[SoundRecorder]) -> None:
        if recorder is not None:
            print("Recorder:", type(recorder).__mro__)
            recorder.sound = self.sound()
        self.update()

    @pyqtSlot(bool)
    # @protect
    def _onButtonRecordClicked(self, checked: bool) -> None:
        recorder = self.soundRecorder()
        if recorder is None:
            LOG.warning("QSoundControl: No recorder, sorry!")
        elif checked:
            LOG.info("QSoundControl: Recording sound")
            recorder.record(self._sound)
        else:
            LOG.info("QSoundControl: Stop recording sound")
            recorder.stop()

    @pyqtSlot(bool)
    # @protect
    def _onButtonPlayClicked(self, checked: bool) -> None:
        player = self.soundPlayer()
        if player is None:
            LOG.warning("QSoundControl: No player, sorry!")
        elif checked:
            LOG.info("QSoundControl: Playing sound %s on player %s",
                     self._sound, player)
            player.play(self._sound, blocking=False)
        else:
            LOG.info("QSoundControl: Stop playing sound on player %s",
                     player)
            player.stop()

    @pyqtSlot(bool)
    # @protect
    def _onButtonLoadClicked(self, checked: bool) -> None:
        fileName, selectedFilter = QFileDialog.getOpenFileName(
            self, "Load audio file", os.getcwd(),
            "Audio Files (*.wav *.mp3)");
        try:
            print("Loading file:", fileName, selectedFilter)
            sound = Sound(fileName)
        except Execption as ex:
            print("Exception:", ex)
            sound = None
        self.setSound(sound)

    @pyqtSlot(bool)
    # @protect
    def _onButtonInfoClicked(self, checked: bool) -> None:
        player = self.soundPlayer()
        recorder = self.soundRecorder()
        if player is None:
            playerText = "None"
        elif player.sound is None:
            playerText = "without sound"
        else:
            playerText = f"with sound (duration={player.sound.duration})"

        if recorder is None:
            recorderText = "None"
        elif recorder.sound is None:
            recorderText = "without sound"
        else:
            recorderText = f"with sound (duration={recorder.sound.duration})"

        print(f"info[QSoundControl]: Sound: {self._sound}, "
              f"Player: {playerText}, Recorder: {recorderText}")
        print(str(player))

    @pyqtSlot(int)
    # @protect
    def _onLoopChanged(self, state: int) -> None:
        player = self.soundPlayer()
        if player is not None:
            player.loop = (state == Qt.Checked)

    @pyqtSlot(int)
    # @protect
    def _onReverseChanged(self, state: int) -> None:
        player = self.soundPlayer()
        if player is not None:
            player.reverse = (state == Qt.Checked)

    def data_changed(self, sound: Sound, info: Sound.Change) -> None:
        # pylint: disable=invalid-name
        LOG.debug("QSoundControl: sound changed: duration=%s",
                  sound.duration)

    def player_changed(self, player: SoundPlayer,
                       info: SoundPlayer.Change) -> None:
        # pylint: disable=invalid-name
        LOG.debug("QSoundControl: player changed: playing=%s, position=%s",
                  player.playing, player.position)
        if info.state_changed:
            self.update()

        if info.position_changed:
            self._soundScrollbar.update()

    def recorder_changed(self, player: SoundRecorder,
                         info: SoundRecorder.Change) -> None:
        # pylint: disable=invalid-name
        if info.state_changed:
            self.update()

    def update(self) -> None:
        sound = self.sound()
        player = self.soundPlayer()
        recorder = self.soundRecorder()
        LOG.debug("QSoundControl.update: sound=%s, player=%s, recorder=%s",
                  sound, player, recorder)
        self._buttonPlay.setEnabled(player is not None and
                                    player.sound is not None)
        self._buttonPlay.setChecked(player is not None and
                                    player.playing)
        self._buttonRecord.setVisible(recorder is not None)
        self._buttonRecord.setEnabled(recorder is not None and
                                      player.sound is not None)
        self._buttonRecord.setChecked(recorder is not None and
                                      recorder.recording)
        self._b1.setVisible(recorder is not None)
        self._b2.setVisible(recorder is not None)

        self._checkboxLoop.setEnabled(player is not None)
        self._checkboxLoop.setCheckState(Qt.Checked if
                                         player is not None and
                                         player.loop else Qt.Unchecked)
        self._checkboxReverse.setEnabled(player is not None)
        self._checkboxReverse.setCheckState(Qt.Checked if
                                            player is not None and
                                            player.reverse else
                                            Qt.Unchecked)
        super().update()


class SoundViewQtAdapter(SoundView):
    """A wrapper class that encapsulates a QSoundControl within
    a Deep Learning Toolbox `SoundView` API.
    """

    _qWidget: QSoundControl = None

    def __init__(self, **kwargs) -> None:
        self._qWidget = QSoundControl()
        super().__init__(**kwargs)

    @property
    def sound(self) -> Optional[Sound]:
        """The :py:class:`Sound` to be displayed in this
        :py:class:`SoundView`.  If `None` the display will
        be cleaned.
        """
        return self._qWidget.sound()

    @sound.setter
    def sound(self, sound: Optional[Soundlike]) -> None:
        self._qWidget.setSound(Sound.as_sound(sound=sound))

    @property
    def player(self) -> SoundPlayer:
        """A :py:class:`SoundPlayer` observed by this
        :py:class:`SoundView`. Activities of the player may be
        reflected by this view and the view may contain
        graphical elements to control the player. The player
        may be `None` in which case such controls will be disabled.
        """
        return self._qWidget.soundPlayer()

    @player.setter
    def player(self, player: Optional[SoundPlayer]) -> None:
        self._qWidget.setSoundPlayer(player)

    @property
    def recorder(self) -> SoundRecorder:
        """A :py:class:`SoundRecorder` observed by this
        :py:class:`SoundView`. Activities of the recorder may be
        reflected by this view and the view may also contain
        graphical elements to control the recorder.
        """
        return self._qWidget.soundRecorder()

    @recorder.setter
    def recorder(self, recorder: Optional[SoundRecorder]) -> None:
        self._qWidget.setSoundRecorder(recorder)

    def qWidget(self) -> QWidget:
        return self._qWidget


class QSoundSeparator(QWidget):
    """A :py:class:`QSoundSeparatorView` supports experimenting with
    Soundseparators, displaying the original (mixed) sound as well
    as the separated output sound.
    """

    def __init__(self, sound: Optional[Sound] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
    
    def _initUI(self) -> None:
        """Initialize the GUI
        """
        self._mixed = QSoundControl(player=SoundPlayer())
        self._separatorControls = QWidget()
        self._outputs = [
            QSoundControl(player=SoundPlayer()),
            QSoundControl(player=SoundPlayer())
        ]

    def _layoutUI(self) -> None:
        mixedLayout = QVBoxLayout()
        mixedLayout.addWidget(self._mixed)

        mixedBox = QGroupBox("Mix")
        mixedBox.setLayout(mixedLayout)

        outputsLayout = QVBoxLayout()
        outputsLayout.addWidget(self._separatorControls)
        for output in self._outputs:
            outputsLayout.addWidget(output)
        
        outputsBox = QGroupBox("Outputs")
        outputsBox.setLayout(outputsLayout)

        layout = QVBoxLayout()
        layout.addWidget(mixedBox)
        layout.addWidget(outputsBox)
        self.setLayout(layout)



class QtSoundDisplay(SoundDisplay, QStandalone):
    """The :py:class:`QtSoundDisplay` implements a Deep Learning
    Toolbox :py:class:`SoundDisplay`, that is a user interface to
    display a sound.

    The :py:class:`QtSoundDisplay` is a subclass of
    :py:class:`QStandalone`, allowing to display the sound as
    a standalone application.
    """

    def __init__(self, view: Optional[View] = None, **kwargs) -> None:
        if view is None:
            view = SoundViewQtAdapter()
        super().__init__(view=view, **kwargs)
    
    def show(self) -> None:
        widget = self.view.qWidget()
        widget.show()
        self.showStandalone()
