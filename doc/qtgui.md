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



# Smooth user experience

The QtGUI aims to be a responsive user interface.  Even if complex
computations are run in the background, the user should still have a
smooth experience. Ideally, background processes, availability of
resources, etc. should be perceivable by the user: it is acceptable if
a background operation requires some time, but then the user interface
should reflect this and let the user that some work is going on.

## Responsiveness

The graphical user interface should always stay responsive. It is not
acceptable if the mouse does not move or if buttons do not react to
mouse clicks.

If some graphical elements can currently not be used because the
backend is performing some heavy operation (like loading data or
modules), these elements should be disabled so that the user knows
that they are currently not available. Showing some busy indicator is
also a good idea.

In general, it should be clearly documented which operations of the
backend block other operations and these dependencies should be
reflected by the graphical user interface.


## Checks

The following list contains some proposals of what to check when
developing a widget:

### Resizing

Try to resize the widget. Resizing requires a repaint of the full
widget. If repainting takes some time, it may be appropriate to only
do a partial repaint during resizing.

Another important aspect can be to outsource complex computation into
a background thread and call `widget.update()` once the computation
is finished. The class `QThreadedUpdate` and the decorator
`pyqtThreadedUpdate` support such a design.
  
