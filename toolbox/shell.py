import sys
from cmd import Cmd
     
class ToolboxShell(Cmd):
    
    prompt = 'dl-toolbox> '
    intro = "Welcome to the dl-toolbox."

    def __init__(self, toolbox=None, **kwargs):
        super().__init__(**kwargs)
        self._toolbox = toolbox

    def emptyline(self):
        """React to empty lines.

        By default when an empty line is entered, the last command is
        repeated. We will disable this behaviour.
        """
        pass

    def do_exit(self, inp):
        """Exit the shell.
        """
        print("Bye")
        return True
       
    do_bye = do_exit
    do_quit = do_exit
    do_EOF = do_exit
    
    def do_list(self, inp):
        """List resources available to the program.
        """
        print("Adding '{}'".format(inp))

    def do_info(self, inp):
        """Display information on the current Toolbox.
        """
        if self._toolbox is None:
            print("No toolbox available")
            return

        self._title("Toolbox:")
        print(str(self._toolbox))

    def _title(self, title):
        print(title)
        print(len(title) * self.ruler)
        print
        
    def do_datasource(self, inp):
        if not self._toolbox:
            print("fetch: error: no Toolbox")
            return

        print(f"Current datsource: {self._toolbox.datasource}")
        print(f"Available datsources:")
        for datasource in self._toolbox._datasources:
            print(f"  {datasource}")
            
    def do_show(self, inp):
        """Show the current input data.
        """
        if self._toolbox is None:
            print("No toolbox available.")
            return

        if self._toolbox.input_data is None:
            print("No input data available.")
            return
        
        import matplotlib.pyplot as plt
        plt.gray()
        plt.imshow(self._toolbox.input_data)

        print("Close window to continue ...")
        plt.show()
        
    def do_modules(self, inp):
        """List the modules that have been loaded.
        """
        for i, m in enumerate(sys.modules):
            print(f"({i}) {m}")

    def do_fetch(self, inp):
        """Fetch input data from the current datasource.
        """
        if not self._toolbox:
            print("fetch: error: no Toolbox")
            return

        if not self._toolbox.datasource:
            print("fetch: error: no datasource.")
            return

        self._toolbox.datasource.fetch()
        print(f"Fetched: {self._toolbox.input_metadata}")

    def do_gui(self, inp):

        if not self._toolbox:
            print("gui: error: no Toolbox")
            return

        if self._toolbox._gui:
            print("gui: error: GUI was already started.")
            return

        print("panel: warning: experimental stuff! "
              "You may observe strange behaviour ...")
        rc = self._toolbox.start_gui(threaded=True, panels=[], tools=[],
                                     networks=[], datasources=[],
                                     focus=False)

    def do_panel(self, inp):
        if not self._toolbox:
            print("panel: error: no Toolbox")
            return

        if not self._toolbox.gui:
            print("panel: error: No GUI was started.")
            return

        print(f"panel: warning: experimental stuff! "
              "You may observe strange behaviour ...")
        try:
            if inp:
                self._toolbox.gui.panel(inp, create=True, show=True)
            else:
                # FIXME[hack]: do not access private members - we need some API
                tabs = self._toolbox.gui._tabs
                print(tabs.tabText(tabs.currentIndex()))
        except KeyError as error:
            print(f"panel: error: {error}")
            # FIXME[hack]: do not access private members - we need some API
            print("panel: info: Valid panels ids are:",
                  [m.id for m in self._toolbox.gui._panelMetas])

    def do_threads(self, inp):
        import threading
        for thread in threading.enumerate():
            print(thread)

if __name__ == '__main__':
    ToolboxShell().cmdloop()

    
