#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""A command line interface for running and sound face tools.

.. moduleauthor:: Ulf Krumnack

"""

# standard imports
import os
from argparse import ArgumentParser

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.base.sound import SoundReader, SoundPlayer


def main():
    """Main program: parse command line options and start sound tools.
    """

    parser = ArgumentParser(description='Deep learning based sound processing')
    parser.add_argument('sound', metavar='SOUND', type=str, nargs='*',
                        help='a SOUND file to use')
    parser.add_argument('--play', action='store_true', default=False,
                        help='play soundfiles')
    ToolboxArgparse.add_arguments(parser)

    args = parser.parse_args()
    ToolboxArgparse.process_arguments(args)

    if args.play:
        print(f"play sound files:")
        reader = SoundReader()
        player = SoundPlayer()
        for soundfile in args.sound:
            print(f"playing '{soundfile}' ... ")
            sound = reader.read(soundfile)
            player.play(sound)
            print(f"... '{soundfile}' finished.")
    else:
        print(f"args.sound={args.sound}")


if __name__ == "__main__":
    main()
