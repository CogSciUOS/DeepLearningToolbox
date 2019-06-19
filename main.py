#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''A framework-agnostic visualisation tool for deep neural networks

.. moduleauthor:: Rüdiger Busche, Petr Byvshev, Ulf Krumnack, Rasmus
Diederichsen

'''


import sys
import argparse


import importlib

class ImportInterceptor(importlib.abc.MetaPathFinder):

    def find_module(self, fullname, path=None):
        print(f"find_module({fullname}, {path})")
        return None

if not hasattr(sys, 'frozen'):
    # sys.meta_path = [ImportInterceptor()] + sys.meta_path
    pass


from toolbox import Toolbox

def main():
    '''Start the program.'''

    toolbox = Toolbox()

    parser = argparse.ArgumentParser(
        description='Visual neural network analysis.')
    toolbox.add_command_line_arguments(parser)
    args = parser.parse_args()

    # This will also run the graphical interface
    rc = toolbox.process_command_line_arguments(args)

    print(f"Main: exiting gracefully (rc={rc}).")
    sys.exit(rc)


if __name__ == '__main__':
    main()
