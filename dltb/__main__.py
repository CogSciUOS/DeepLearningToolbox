"""Top level script environment [1] for the Deep Learning Toolbox
module (`dltb`).

Background
----------

The file `__main__.py` is read when the module is executed as a
script.  This can happen in two different situations: the usual way
is to pass the `-m dltb` switch to the python interpreter

  python -m dltb

Then the module `dltb` is imported (its `__init__.py` file is
executed) and afterwards this file (`__main__.py`) is imported and
executed.  In this situation, the package (`__package__`) is `'dltb'`
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

def main():
    """The main program to be executed.
    """
    with_dltb_module = 'dltb' in sys.modules
    with_m_switch = bool(__package__)
    with_m_switch2 = __file__ == sys.argv[0]

    print("Deep Learning Toolbox (dltb):", *sys.argv)
    print(f"name: '{__name__}'")  # always '__main__'
    print(f"package: '{__package__}'")  # 'package' with -m and '' otherwise
    print(f"dltb was imported: {with_dltb_module}")
    print(f"the '-m' switch was used: {with_m_switch}/{with_m_switch2}")


if __name__ == '__main__':  # should always be True
    main()
else:
    # This should not happen when the module is loaded in the standard way
    # (by calling 'python -m dltb' or 'python dltb').
    # Only strange usage should result in getting lost here, e.g.
    #  * importing the module directly: from dltb import __main__
    # In such situations, we will inform the user ...
    print("To run the Deep Learning Toolbox (dltb), call 'python -m dltb' ",
          file=sys.stderr)
    print("or use the script `dl-toolbox.py`.",
          file=sys.stderr)
