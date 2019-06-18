import traceback

def handle_exception(exception: BaseException):
    print(exception)
    print(f"\nUnhandled exception ({type(exception).__name__}): {exception}")
    traceback.print_tb(exception.__traceback__)
    print("\n")
