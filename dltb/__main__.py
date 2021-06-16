"""Top level script environment [1] for the Deep Learning Toolbox
module (`dltb`).

Background
----------

The file `__main__.py` is read when the module is executed as a
script.  This can happen in two different situations: the usually way
is to pass the `-m dltb` switch to the python interpreter

  python -m dltb

Then the module `dltb` is imported (its `__init__.py` file is
executed) and afterwards this file (`__main__.py`) is imported and
executed. In this situation, the package (`__package__`) is '`dltb`'
and the first command line argument (`sys.argv[0]`) is set to the
absolute path pointing to this file (`'.../dltb/__main__.py'`).

This file is also executed when calling

  python dltb

However, in this situation, the `dltb` module is not automatically
imported, and the first argument (`sys.argv[0]`) is set to `'dltb'`.

[1] https://docs.python.org/3/library/__main__.html

"""
import sys

# make sure that the dltb package is imported, even if this file
# was invoked without the -m switch.
#from dltb import config

with_dltb_module = 'dltb' in sys.modules
with_m_switch = bool(__package__)
with_m_switch2 = __file__ == sys.argv[0]


def main():
    print("Deep Learning Toolbox (dltb):", *sys.argv)
    print(f"name: '{__name__}'")  # always '__main__'
    print(f"package: '{__package__}'")  # 'package' with -m and '' otherwise
    print(f"dltb was imported:", 'dltb' in sys.modules)
    print(f"the '-m' switch was used:", with_m_switch, with_m_switch2)


if __name__ == '__main__':  # should always be True
    main()
