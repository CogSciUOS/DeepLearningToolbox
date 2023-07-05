#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A framework-agnostic visualisation tool for deep neural networks

.. moduleauthor:: Rüdiger Busche, Petr Byvshev, Ulf Krumnack, Rasmus
Diederichsen

"""

# standard imports
import sys

# FIXME[old]: rework Toolbox!
from toolbox import Toolbox


def main():
    """Start the program."""

    toolbox = Toolbox()
    toolbox.process_command_line_arguments()

    # This will also run the graphical interface
    return_code = toolbox.run()

    print(f"Main: exiting gracefully (return code={return_code}).")
    sys.exit(return_code)


if __name__ == '__main__':
    main()
