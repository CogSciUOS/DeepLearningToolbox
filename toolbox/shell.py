import sys
from cmd import Cmd
     
class ToolboxShell(Cmd):
    
    prompt = 'dl-toolbox> '
    intro = "Welcome to the dl-toolbox."

    def __init__(self, toolbox=None, **kwargs):
        super().__init__(**kwargs)
        self._toolbox = toolbox
    
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
        print("datasource", inp)

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

if __name__ == '__main__':
    ToolboxShell().cmdloop()

    
