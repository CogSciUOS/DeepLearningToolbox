Notes for the developer
=======================

Coding guidelines
-----------------
The following set of rules have been arbitrarily chosen by me (Rasmus) and my
linter.

#. Do not exceed 100 characters per line, unless forced to at gunpoint
#. Use spaces instead of tabs

    #. 1 tab shall expand to 4 spaces

#. Use ``'`` single quotes ``'`` instead of ``"`` double quotes ``"``
#. prefix private attributes and methods with ``_``
#. Use camelCase instead of snake_case
#. Use ``numpydoc`` sphinx syntax for documentation


Planned architecture
--------------------
The following layout was planned::

    +-----------+
    |MODEL      |
    +-----------+---------------------------------------------------------------+
    |                                                                           |
    |        +-----------------------+              +----------------------+    |
    |        | Network               |              |  Stimulus            |    |
    |        +-----------------------+              +----------------------+    |
    |        |                       | <------------+ N, H, W, C           |    |
    |        | getActivations(layer, |              | intensities : ndarray|    |
    |        |  stimulus)            |              |                      |    |
    |        |                       |              +------------------+---+    |
    |        | ...                   |                                 ^        |
    |        |                       |                                 |        |
    |        +-----------+-----------+                                 |        |
    |                    ^                                             |        |
    |                    | getActivations(stimulus, layer)             |        |
    |                    |                                             |        |
    |      +-------------+---------------+                             |        |
    |      |       ShapeAdaptor          |                             |        |
    |      +-------------+---------------+   +-------------------+     |        |
    |                    |                   |    Observable     |     |        |
    |         +----------+-----------+       +-------------------+     |        |
    |         |     Visualization    |       | registerObserver()|     |        |
    |         +----------------------+       |                   |     |        |
    |         |                      +-----> |                   |     |        |
    |         |  current stimulus    |       +-------------------+     |        |
    |         |  current unit        |                                 |        |
    |         |  current layer       +---------------------------------+        |
    |         |  cache               |                                          |
    |         |  overlay             |                                          |
    |         |  registerObserver()  |                                          |
    |         |                      | <-------------------------+              |
    |         +------+---------+-----+                           |              |
    |                ^         ^                                 |              |
    |                |         +------------+                    |              |
    |                |                      |                    |              |
    |   +------------+---------+    +-------+------------+    +--+------------+ |
    |   |     Occlusion        |    |  Heatmap           |    |  Activation   | |
    |   +----------------------+    +--------------------+    +---------------+ |
    |                                                                           |
    |                                                                           |
    |                                                                           |
    |                                                                           |
    |                                                                           |
    +---------------------------------------------------------------------------+


    +---------------+
    |CONTROLLER     |
    +---------------+-----------------------------------------------------+
    |                                                                     |
    |             +---------------------+                                 |
    |             |     Controller      |                                 |
    |             +---------------------+                                 |
    |             |                     |                                 |
    |             |  unitSelected()     |                                 |
    |             |  layerSelected()    |                                 |
    |             |  inputChanged()     |                                 |
    |             |  visChanged()       |                                 |
    |             |                     |                                 |
    |             |  currentNet         |                                 |
    |             |  currentVis         |                                 |
    |             |                     | <---------+                     |
    |             +----+----------------+           |                     |
    |                  ^                            |                     |
    |                  |                            |                     |
    |    +-------------+----------+            +----+-----------------+   |
    |    |    GUIController       |            |   CMDLineController  |   |
    |    +------------------------+            +----------------------+   |
    |    |                        |            |                      |   |
    |    |   unitClicked          |            |        ??            |   |
    |    |   layerClicked         |            |                      |   |
    |    |   nextClicked          |            |                      |   |
    |    |   ...                  |            |                      |   |
    |    |   keyPressed           |            |                      |   |
    |    +------------------------+            +----------------------+   |
    |                                                                     |
    |                                                                     |
    |                                                                     |
    +---------------------------------------------------------------------+

Current architecture
--------------------
Currently, the state of the app is this:
The basic idea is as follows.

The model
^^^^^^^^^
The model class in ``model/model.py`` contains all of the application's state, for instance

* Current activations
* Current layer
* Currently selected unit
* Currently active network

All actions made by the user will ultimately alter the model's state. All GUI elements act as an ``observer`` of a shared model, which means they register themselves and have the model notify them of any changes. They must then fetch relevant data from the model and repaint themselves with the updated data. They ideally do not contain any state themselves and are simply views onto the model and thus more easily replaceable with other kinds of interfaces.

The controller
^^^^^^^^^^^^^^
The controller handles triggering the actual updates of the model. GUI elements relay events to their controller which then decides how to update the model accordingly. All complex controls (hotkeys, input event combinations, etc.) can thus be kept separate from the user interface. Nevertheless, a controller class is typically tailored to a user interface type, so that replacing the gui with e.g. a CL interface will often necessitate the implementation of a new controller.
The controller lives in ``controller/activationscontroller.py``. It is currently envisioned that all the widgets could share the same controller and that widget containers (such as the main window) add their children's controllers as child controllers. Every controller would then be able to call into its parent in order to relay events triggered from within to the top level gui elements. However, this functionality is currently unused.

View
^^^^
All GUI elements are views onto the model. They accept a controller object in ``setController`` which in turn knows its model. They provide a ``modelChanged`` callback for responding to changes.
