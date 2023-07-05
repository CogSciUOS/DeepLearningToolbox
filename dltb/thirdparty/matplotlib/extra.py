import matplotlib
#matplotlib.rcParams["toolbar"] = "toolmanager"
import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolToggleBase

class SelectButton(ToolToggleBase):
    default_toggled = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def enable(self, event):
        pass

    def disable(self, event):
        pass

    def __init__old(self):
        # The backend can be changed by mpl.use('...')
        # possible values depending on the matplotlit installation, e.g.,
        # 'agg', Qt5Agg, 'TkAgg', 'wxAgg '
        backend = mpl.get_backend()

        try:
            window = manager.window
        except AttributeError:
            window = canvas.window()
        
        if backend == 'TkAgg':
            window.wm_iconbitmap('assets/logo.png')  # 'icon.ico'
        elif backend == 'wxAgg':
            import wx
            window.SetIcon(wx.Icon('assets/logo.png', wx.BITMAP_TYPE_ICO))
        elif backend == 'Qt5Agg':
            from PyQt5 import QtGui
            window.setWindowIcon(QtGui.QIcon('assets/logo.png'))

            toolbar = self._figure.canvas.toolbar
            toolbar.clear()
            print("Cleared the toolbar!")

            from PyQt5 import QtWidgets
            action = QtWidgets.QAction(QtGui.QIcon('assets/logo.png'), "tool")
            toolbar.addAction(action)
            print("Added new action to toolbar!")

            toolbar.addWidget(QtWidgets.QPushButton("Hallo"))
            print("Added new button to toolbar!")

            # toolbar = window.findChild(QtWidgets.QToolBar)
            # toolbar.setVisible(False)



import matplotlib.pyplot as plt
plt.rcParams['toolbar'] = 'toolmanager'
from matplotlib.backend_tools import ToolBase

class ListTools(ToolBase):
    """List all the tools controlled by the `ToolManager`."""
    # keyboard shortcut
    default_keymap = 'm'
    description = 'List Tools'
    #image = 'back'
    # 'back', 'filesave', 'forward', 'hand', 'help', 'home', 'matplotlib',
    # 'move', 'qt4_editor_options', 'subplots', 'zoom_to_rect'
    image = 'matplotlib'
    
    # specifying a filename for some reason does not work. Have a closer
    # look at the sourcecode to see what may cause the problem ...
    # /space/conda/user/ulf/envs/dl-toolbox/lib/python3.8/site-packages/matplotlib/backend_bases.py
    #image = 'assets/logo.png'
    #image = '/home/ulf/projects/github/DeepLearningToolbox/assets/logo.png'

    def trigger(self, *args, **kwargs):
        print("self.image=", self.image)
        print('_' * 80)
        print("{0:12} {1:45} {2}".format(
            'Name (id)', 'Tool description', 'Keymap'))
        print('-' * 80)
        tools = self.toolmanager.tools
        for name in sorted(tools):
            if not tools[name].description:
                continue
            keys = ', '.join(sorted(self.toolmanager.get_tool_keymap(name)))
            print("{0:12} {1:45} {2} [{3}]".format(
                name, tools[name].description, keys, tools[name].image))
        print('_' * 80)
        print("Active Toggle tools")
        print("{0:12} {1:45}".format("Group", "Active"))
        print('-' * 80)
        for group, active in self.toolmanager.active_toggle.items():
            print("{0:12} {1:45}".format(str(group), str(active)))




class Experiments:

    def __init__(self, toolbar: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._figure = plt.figure(figsize=(8, 6))
        #
        # Extend the toolbar (using the "toolmanager" interface)
        #
        canvas = self._figure.canvas
        manager = canvas.manager

        manager.toolmanager.remove_tool('forward')
        manager.toolmanager.add_tool('List', ListTools)
        manager.toolbar.add_tool('List', ListTools)

    #
    # Toolbar
    #

    def add_toolbar_button(self) -> None:
        """Add a toolbar button to this Plotter.
        """
        if mpl.rcParams["toolbar"] == "toolmanager":
            self._add_toolbar_button_with_toolmanager()

    def _add_toolbar_button_with_toolmanager(self) -> None:
        # Add a toolbar button using toolmanager
        #matplotlib.rcParams["toolbar"] = "toolmanager"
        toolmanager = self._figure.canvas.manager.toolmanager
        toolmanager.add_tool('Toggle recording', ToolToggleBase)
        self.my_select_button = toolmanager.get_tool('Toggle recording')
        self._figure.canvas.manager.toolbar.\
            add_tool(self.my_select_button, "toolgroup")


    def _hide_toolbar(self):
        # does not work for Qt5Agg backend ...
        
        canvas = self._figure.canvas
        #canvas.toolbar_visible = toolbar
        #canvas.header_visible = toolbar
        #canvas.footer_visible = toolbar
