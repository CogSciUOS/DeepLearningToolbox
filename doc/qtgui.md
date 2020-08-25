# The Deep Learning ToolBox Qt Graphical User Interface (QtGUI)

The QtGUI is a collection of Widgets that can be used for graphically
interfacing individual components of the Deep Learning ToolBox. It is
based on the Qt5 library for Python (PyQt5).


# Coding standards for the Qt graphical user interface

In the Qt graphical user interface we deviate in some points from the
[general coding standards of the Deep Learning ToolBox](coding.md)
to conform with the Qt standards:
* use camel case instead of snake case for methods and attribute names

# Linting the QtGUI

```sh
pylint --extension-pkg-whitelist=PyQt5 --method-naming-style=camelCase --attr-naming-style=camelCase --attr-naming-style=camelCase [FILE]...
```

