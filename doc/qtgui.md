# The Deep Learning ToolBox Qt Graphical User Interface (QtGUI)

The QtGUI is a collection of Widgets that can be used for graphically
interfacing individual components of the Deep Learning ToolBox. It is
based on the Qt5 library for Python (PyQt5).



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



# Coding standards for the Qt graphical user interface

In the Qt graphical user interface we deviate in some points from the
[general coding standards of the Deep Learning ToolBox](coding.md)
to conform with the Qt standards:
* use camel case instead of snake case for methods and attribute names
* Name classes that are derived from `QObject` starting with a
  capital `Q`, e.g., `QMyWidget`. Notice that it is problematic
  to derive a Python class from two `QObjects` (at least some additional
  care hase to be taken - I still have to work out the details). So as
  a rule of thumb, a class should have at most one `Q`-class as 
  base class, and if it does so, it will also be a `Q`-class.

## Problematic aspects

### Threading

Qt uses real threads allowing for real parallelism, in contrast to
Python that just does its own preemptive multitasking.  This speeds
can make the interface smoother, as calculations can really be done in
the background, but it puts some extra burden to communication between
threads.

### Multiple inheritance

It seems not to be possible to do multiple inheritance, inheriting
from different native Qt classes (but I have not really systematically
checked this yet).


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



# Linting the QtGUI

As the QtGUI adopts different coding standards than the rest of the
project, `pylint` has to be invoked with some special arguments:

```sh
pylint --rcfile=qtgui/pylintrc [FILE]...
```





# Some components

## Data view

* QImageView, QSoundView, etc.: data observer, just display

* QDataView: combines different views, again data observer, but may
  also observe toolbox or datafetcher

* QDatasourceNavigate: contains a QDataView, but is also
  Toolbox or Datafetch observer


# User Interface Demo

The `qtgui` comes with a demo script `demo.py` which allows to run
individual components in demo mode.  The script can be invoked in
different ways:

From the command line:
```sh
python qtgui/demo.py
```
or
```sh
python -m qtgui.demo
```

It is also possible to run the demo interactively, e.g., from an
IPython shell:
```
import qtgui.debug
from qtgui.demo import QDemo

demo = QDemo()
demo.show()
```

