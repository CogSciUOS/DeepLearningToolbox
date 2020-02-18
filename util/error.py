import traceback

import sys

def handle_exception(exception: BaseException):
    print(exception)
    print(f"\nUnhandled exception ({type(exception).__name__}): {exception}")
    # sys.stderr may by set to os.devnull ...
    #traceback.print_tb(exception.__traceback__, file=sys.stderr)
    traceback.print_tb(exception.__traceback__, file=sys.stdout)
    print("\n")
