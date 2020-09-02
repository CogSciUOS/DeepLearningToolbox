# Logging

## Introducing logging into the code

Logging is based on the Python 
[`logging` library](https://docs.python.org/3/library/logging.html).
Logging on different levels can be done using a 
[`Logger` object](https://docs.python.org/3/library/logging.html#logger-objects)
and its methods `debug`, `info`, `warning`, `error`, and `critial`.

For performance reasons, it is recommended to not use f-strings or
other types of string constructions that unconditionally do the
construction, as this may cause some overhead not needed if the
logging is going to be ignored. Instead the 
[Python format string syntax]([Format String Syntax](https://docs.python.org/3/library/string.html#formatstrings) with its
[Format Specification Mini-Language](https://docs.python.org/3/library/string.html#format-specification-mini-language) should be applied.
  

