#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""Sound experiments.

This was in large parts based on the examples from the
`sounddevice` documentation at:

  * https://python-sounddevice.readthedocs.io/en/0.4.0/examples.html

However the examples have been heavily refactored to better fit into
the Deep Learning ToolBox. As a result, by now some parts of this
experiment have already been integrated into the Deep Learning ToolBox
in form of the following classes:

  * :py:class:`dltb.thirdparty.soundfile.SoundReader`
  * :py:class:`dltb.thirdparty.soundfile.SoundWriter`
  * :py:class:`dltb.thirdparty.sounddevice.SoundPlayer`
  * :py:class:`dltb.thirdparty.sounddevice.SoundRecorder`
  * :py:class:`qtgui.widgets.sound.QSoundDisplay`

These classes are implementations of the abstract classes from the
toolbox base.

  * :py:class:`dltb.base.sound.Sound`
  * :py:class:`dltb.base.sound.SoundReader`
  * :py:class:`dltb.base.sound.SoundWriter`
  * :py:class:`dltb.base.sound.SoundPlayer`
  * :py:class:`dltb.base.sound.SoundRecorder`
  * :py:class:`dltb.base.sound.SoundDisplay`

Some other parts have not been integrated yet:

  * The matplotlib interface

What still remains is the main program that shows how the classes
from the Toolbox can be used.


.. moduleauthor:: Ulf Krumnack


Invocation
----------

dl-sound.py

    Play the demo sound file ()

--list-devices

   Display a list of available devices and exit


Providing a sound sample

    --record: record a sound sample (2 seconds)
    --generate: generate an audiosample

    If none of the above applies, the file referred to by
    `DEMO_SOUNDFILE` will be read in.


Graphical User Interfaces
-------------------------

    --qtgui

    --matplotlib

    --matplotlib-old

"""

# FIXME[bug]:
#  - recoding somehow sets player sound to None

# standard imports
from typing import Callable
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import sys
import os

# third party imports
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import QApplication


# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.base.sound import Sound, SoundDisplay
from dltb.base.sound import SoundReader, SoundPlayer, SoundRecorder
from dltb.util.error import print_exception
from qtgui.widgets.sound import QSoundControl, QSoundSeparator



# Configuration

# DEMO_SOUNDFILE: a file to be used for the demo
for directory in (os.environ.get('HOME', False),
                  os.environ.get('NET', False)):
    if not directory:
        continue
    DEMO_SOUNDFILE = os.path.join(directory, 'projects', 'examples',
                                  'mime', 'audio', 'wav',
                                  'le_tigre.wav')
    if os.path.isfile(DEMO_SOUNDFILE):
        break
else:
    DEMO_SOUNDFILE = None


# ------------------------------------------------------------------------------
# matplotlib
# ------------------------------------------------------------------------------

from typing import List
import queue

from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class MatplotlibSoundPlotter(SoundDisplay):
    """Plot a sound wave using MatPlotLib.

    Plot the live microphone signal(s) with matplotlib.

    Matplotlib and NumPy have to be installed.

    """

    def __init__(self, samplerate: float,
                 window: float = 200, channels: List[int] = [1],
                 downsample: int = 10, interval: float = 30,
                 standalone: bool = True,
                 figure: Figure = None, ax=None) -> None:
        """
        samplerate: float
            sampling rate of audio device
        channels: List[int]
            input channels to plot (default: the first)
        window: float
            visible time slot (in ms)
        interval:
            minimum time between plot updates (in ms)
        downsample: int
            display every Nth sample

        ax:
        """
        self._queue = queue.Queue()
        self._running = False
        self._animation = None

        self._interval = interval
        self._downsample = downsample
        self._standalone = standalone

        # Channel numbers start with 1
        self._mapping = [c - 1 for c in channels]

        length = int(window * samplerate / (1000 * downsample))
        self._plotdata = np.zeros((length, len(channels)))

        # create the figure
        if standalone:
            self._fig, ax = plt.subplots()
            # self._fig = Figure(figsize=(5,5), dpi=100)
            # ax = self._fig.add_subplot(111)
        else:
            self._fig = figure

        self._lines = ax.plot(self._plotdata)
        if len(channels) > 1:
            ax.legend(['channel {}'.format(c) for c in channels],
                      loc='lower left', ncol=len(channels))
        ax.axis((0, len(self._plotdata), -1, 1))
        # ax.set_yticks([0])
        ax.yaxis.grid(True)
        ax.tick_params(bottom=False, top=False, labelbottom=False,
                       right=False, left=False, labelleft=False)

        if self._standalone and self._fig is not None:
            self._fig.tight_layout(pad=0)

    def update_plot(self, frame):
        """This is called by matplotlib for each plot update.

        Typically, audio callbacks happen more frequently than plot
        updates, therefore the queue tends to contain multiple blocks
        of audio data.

        """
        while True:
            try:
                data = self._queue.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            self._plotdata = np.roll(self._plotdata, -shift, axis=0)
            self._plotdata[-shift:, :] = data
        for column, line in enumerate(self._lines):
            line.set_ydata(self._plotdata[:, column])
        return self._lines

    def start_plot(self) -> None:
        if self._animation is None:
            # No animation yet - we will create a new one.
            self._running = True

            # self._canvas = FigureCanvas(self._fig)
            # self._canvas.show()

            # Creating an animation will also start the event loop
            # in its own Thread.
            self._animation = FuncAnimation(self._fig, self.update_plot,
                                            interval=self._interval,
                                            blit=True)
            if self._standalone is not None:
                plt.show()

        elif not self._running:
            # We already have an animation object but it is currently not
            # running. We will start it.
            self._running = True
            if self._animation.event_source is not None:
                self._animation.event_source.start()

    def stop_plot(self) -> None:
        if self._animation is not None and self._running:
            # Stop the currently running animation (but keep the animation
            # object, allowing to restart the animation again later)
            if self._animation.event_source is not None:
                self._animation.event_source.stop()
            self._running = False


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    print("update:", indata.min(), indata.max())
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])



# ------------------------------------------------------------------------------
# function
# ------------------------------------------------------------------------------

def sample_function(function: Callable[[float], float],
                    wavelength: float = 2*np.pi,
                    frequency: float = 440,
                    duration: float = 1.0,
                    samplerate: float = 44100) -> np.ndarray:
    """Create an array of values sampled from a (periodic) function.

    Arguments
    ---------
    function:
        The (periodic) function to use.
    wavelength:
        Wavelength of the function (default 2 pi).
    frequency:
        The function will be adapted to the given frequency (in Hertz).
        For example a value of 440 means that the wave will be played
        at 440 Hz (Stuttgart ptich).
    duration:
        The length of the generated sample.
    samplerate:
        The samplerate at which points are taken from the curve. This
        should be at least twice the frequency.
    """
    samplepoints = np.linspace(0, duration * wavelength * frequency,
                               int(samplerate * duration))
    return function(samplepoints)



# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------



def main():
    """Start the program.
    """
    def int_or_str(text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    # parser = ArgumentParser(add_help=False)
    # args, remaining = parser.parse_known_args()

    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter
        # parents=[parser]
    )

    # General paramters
    parser.add_argument(
        '-c', '--channels', type=int, default=[1], nargs='*',
        metavar='CHANNEL',
        help="input channels to plot (default: the first)")
    parser.add_argument(
        '-r', '--samplerate', type=float, default=44100,
        help="sampling rate of audio device")

    # player/recorder specific paramters
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help="show list of audio devices and exit")
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help="input device (numeric ID or substring)")
    parser.add_argument(
        '-b', '--blocksize', type=int,
        help="block size (in samples)")
    parser.add_argument(
        '--record',  action='store_true',
        help="record a sound")

    # display specific paramters
    parser.add_argument(
        '--qtgui',  action='store_true',
        help="Run the Qt-based graphical user interface")
    parser.add_argument(
        '--matplotlib',  action='store_true',
        help="Run the matplotlib-based graphical user interface")
    parser.add_argument(
        '--matplotlib-old',  action='store_true',
        help="Run the old matplotlib-based graphical user interface")
    parser.add_argument(
        '-n', '--downsample', type=int, default=10, metavar='N',
        help="display every Nth sample (default: %(default)s)")
    parser.add_argument(
        '-i', '--interval', type=float, default=30,
        help="minimum time between plot updates (default: %(default)s ms)")
    parser.add_argument(
        '-w', '--window', type=float, default=200, metavar='DURATION',
        help="visible time slot (default: %(default)s ms)")
    parser.add_argument(
        '-s', '--separate', action='store_true',
        help="Perform sound separation")

    # generator specific arguments
    parser.add_argument(
        '-g', '--generate', nargs='?', const='sine',
        choices=('sine', 'triangle'),
        help="generate a wave (default: %(default)s)")

    ToolboxArgparse.add_arguments(parser)

    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)
    print(vars(args))

    if args.list_devices:
        print(sd.query_devices())
        # old?
        #device_info = sd.query_devices(args.device, 'input')
        # old
        #print("Device info [input]:")
        #print("  Default samplerate:", device_info['default_samplerate'])
        parser.exit(0)

    #
    # Sanity checks
    #
    if any(c < 1 for c in args.channels):
        parser.error('argument CHANNEL: must be >= 1')

    sound = None

    #
    # Sound recorder demo
    #
    if args.record:
        print("Creating recorder")
        recorder = SoundRecorder()

        sound = recorder.record(duration=2, samplerate=args.samplerate)

    #
    # Sound generation demo
    #
    elif args.generate is not None:
        frequency = 440
        duration = 2

        print("Creating sound")
        sound = Sound()
        print(str(sound))
        print("Appending sound")
        if args.generate == 'triangle':
            def triangle(x: float) -> float:
                return np.mod(x, 2)-1
            sound += sample_function(triangle, wavelength=2,
                                     frequency=frequency, duration=duration)
        else:  # args.generate == 'sine':
            sound += sample_function(np.sin, wavelength=2*np.pi,
                                     frequency=frequency, duration=duration)

    #
    # SoundReader demo
    #
    else:
        print("Reading sound2")
        reader = SoundReader()

        soundfile = DEMO_SOUNDFILE
        if soundfile is None:
            print("error: no soundfile provided")
            sys.exit(1)
        sound = reader.read(soundfile)

    #
    # do something with the sound
    #

    #
    # Qt Application
    #
    if args.qtgui:
        print("Creating recorder")
        recorder = SoundRecorder()
        print("Creating player")
        player = SoundPlayer()

        print("Creating QApplication")
        app = QApplication(sys.argv)
        print("Creating QSoundControl")
        window = QSoundControl(sound, player=player, recorder=recorder)
        print("Showing QSoundControl")
        window.show()

        print("Running the event loop")
        app.exec_()
        print("Done.")
        sys.exit(0)

    elif args.matplotlib:
        print("Creating recorder")
        recorder = SoundRecorder()

        print("Creating plotter")
        plotter = MatplotlibSoundPlotter(samplerate=sound.samplerate)

        print("Recording sound")
        # FIXME[bug]: record() got an unexpected keyword argument 'plotter'
        recorder.record(sound, plotter=plotter)

        print(f"Recorded sound: {sound}")

        sys.exit(0)

    elif args.matplotlib_old:
        # FIXME[old]: we need a samplerate concept (config/device/...)
        if args.samplerate is None:
            device_info = sd.query_devices(args.device, 'input')
            args.samplerate = device_info['default_samplerate']

        length = int(args.window * args.samplerate / (1000 * args.downsample))
        plotdata = np.zeros((length, len(args.channels)))

        fig, ax = plt.subplots()
        lines = ax.plot(plotdata)
        if len(args.channels) > 1:
            ax.legend(['channel {}'.format(c) for c in args.channels],
                      loc='lower left', ncol=len(args.channels))
        ax.axis((0, len(plotdata), -1, 1))
        # ax.set_yticks([0])
        ax.yaxis.grid(True)
        ax.tick_params(bottom=False, top=False, labelbottom=False,
                       right=False, left=False, labelleft=False)
        fig.tight_layout(pad=0)

        stream = sd.InputStream(
            device=args.device, channels=max(args.channels),
            samplerate=args.samplerate, callback=audio_callback)

        # FIXME[bug]: name 'update_plot' is not defined
        ani = FuncAnimation(fig, update_plot, interval=args.interval,
                            blit=True)
        with stream:
            plt.show()

    elif args.separate:
        # Perform sound separation

        print("Creating QApplication")
        app = QApplication(sys.argv)
        print("Creating QSoundControl")
        window = QSoundSeparator(sound)
        print("Showing QSoundControl")
        window.show()
        
        print("Running the event loop")
        app.exec_()
        print("Done.")
        sys.exit(0)

    #
    # Sound player demo
    #
    elif sound is not None:
        print("Sound:", str(sound))

        print("Creating player")
        player = SoundPlayer()

        player.play(sound)
        sys.exit(0)




def main_old():
    """Main program: parse command line options and start sound tools.
    """

    parser = ArgumentParser(description='Deep learning based sound processing')
    parser.add_argument('sound', metavar='SOUND', type=str, nargs='*',
                        help='a SOUND file to use')
    parser.add_argument('--play', action='store_true', default=False,
                        help='play soundfile(s)')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the soundfile')

    args = parser.parse_args()

    if args.play:
        print("play sound files:")
        reader = SoundReader()
        player = SoundPlayer()
        for soundfile in args.sound:
            print(f"playing '{soundfile}' ... ")
            sound = reader.read(soundfile)
            player.play(sound)
            print(f"... '{soundfile}' finished.")

    elif args.show:
        print("show sound files:")
        reader = SoundReader()
        player = SoundPlayer()
        display = SoundDisplay(player=player)
        for soundfile in args.sound:
            print(f"displaying '{soundfile}' ... ")
            sound = reader.read(soundfile)
            display.show(sound)
            print(f"... '{soundfile}' finished.")
    else:
        print(f"args.sound={args.sound}")


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        # parser.exit(type(e).__name__ + ': ' + str(e))
        print_exception(exception)
        sys.exit(1)
