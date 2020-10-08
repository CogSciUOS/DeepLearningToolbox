






# Using `RegisterClass`es in the QtGUI

There are several utility classes provided in the `qtgui` module to 
support the use of of `RegisterClass`



                        QNetworkComboBox
                         /           \
                        /             \
               NetworkAdapter          \
                     |                  \
                     |         QRegisterComboBox
               ToolboxAdapter   /         \
                     |         /           \
                     |        /     QAdaptedComboBox
                   RegisterAdapter    /         \
                          \          /           \
                           \        /       [QtWidgets.QComboBox]
                          ItemAdapter


* `qtgui.adapter.ItemAdapter`:
  interface to store items in a list. For display a `itemToText`
  method should be registered.

* `qtgui.adapter.QAdaptedComboBox`:
  an `ItemAdapter` realized as a `QComboBox`. Items added to the
  adapter will show up in the `QComboBox`.

* `qtgui.widgets.register.RegisterAdapter(ItemAdapter)`:
  an auxiliary class that should help to keep the items in an
  `ItemAdapter` in sync with an underlying `Register`


* `qtgui.widgets.register.QRegisterComboBox(RegisterAdapter,QAdaptedComboBox)`:
  a QComboBox that is in sync with an underlying `Register`.

* `qtgui.widgets.register.ToolboxAdapter(RegisterAdapter)`
  an auxiliary class that can be used in two modes: (a) in `Register`
  register mode or (b) in `Toolbox` mode.

* `qtgui.widgets.network.NetworkAdapter(ToolboxAdapter)`
  an `ItemAdapter` for `Network`s. Items of this Adapter are of type
  `InstanceRegisterEntry`, representing instances of the `Network`
  class. Adds a getter `network()` and a setter `setNetwork()`

* `qtgui.widgets.network.QNetworkComboBox(NetworkAdapter, QRegisterComboBox)`:
