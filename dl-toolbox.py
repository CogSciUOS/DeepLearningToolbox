#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''A framework-agnostic visualisation tool for deep neural networks

.. moduleauthor:: Rüdiger Busche, Petr Byvshev, Ulf Krumnack, Rasmus
Diederichsen

'''


import os
import sys
import argparse


import importlib.abc

import inspect

# Tracing the imported modules:
# - MetaPathFinder: goes first and can override builtin modules
# - PathEntryFinder: specifically for modules found on sys.path
class ImportInterceptor(importlib.abc.MetaPathFinder):

    def find_module(self, fullname, path=None):
        #if not '.' in fullname:
        if fullname == 'tensorflow':
            # each line in the stack consists of the following six fields:
            # [0] <class 'frame'>:
            # [1] <class 'str'>: file name
            # [2] <class 'int'>: line number
            # [3] <class 'str'>: the function
            # [4] <class 'list'>: the lines
            # [3] <class 'int'>: ?
            stack = inspect.stack()
            cwd = os.getcwd()
            for i,s in enumerate(stack):
                if s[4] and "import " in s[4][0]:
                    file = s[1]
                    if file.startswith(cwd):
                        file = '.' + file[len(cwd):]
                    if file.startswith('.'):
                        print(f"[{i}/{len(stack)}] {file}:{s[2]}: {s[4][0]}", end='')
                    #break
            print(f"-> find_module({fullname}, {path})")
        return None

# Is the application started from source or is it frozen (bundled)?
# The PyInstaller bootloader adds the name 'frozen' to the sys module:
if not hasattr(sys, 'frozen') and False:
    sys.meta_path = [ImportInterceptor()] + sys.meta_path


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
