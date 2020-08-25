
# generic imports
import os
import sys
import itertools
from cmd import Cmd
from argparse import ArgumentParser, Namespace
from functools import wraps
from typing import Iterable


# toolbox imports
from dltb.base.sound import Sound, SoundDisplay
from dltb.thirdparty.soundfile import SoundReader as SoundfileReader
from dltb.thirdparty.sounddevice import SoundPlayer as SoundDevicePlayer
from toolbox import Toolbox
from tools import Tool
from network import Network
from datasource import Datasource

from util.error import handle_exception

# FIXME[bug]: a (syntax) error in this file lets the toolbox crash!

# FIXME[bug]: an exception during some do_... method lets the toolbox crash!

# FIXME[bug]: completion does not work for words containing dashes
# (e.g. 'mnist-train')

# FIXME[todo]: distinguish between ending the shell (bye/EOF)
# and ending the program (quit/exit)
# Maybe also introduce Ctrl+C (SIGING) to start a shell and
# use Ctrl + Y / Ctrl + \ (SIGQUIT) for quitting the program


class ShellArgumentParserExitException(Exception):
    """A exception raised when a :py:class:`ShellArgumentParser`'
    exit method is called.

    Attributes
    ----------
    status: int
        The exit code (as provided by the status argument to the
        exit method).
    """

    def __init__(self, status: int, **kwargs) -> None:
        """Initialize a :py:class:`ShellArgumentParserExitException`.
        """
        super().__init__(**kwargs)
        self.status = status

class ShellArgumentParser(ArgumentParser):
    """The :py:class:`ShellArgumentParser` is a subclass of python's
    :py:class:`ArgumentParser` that is adapted for use in a
    toolbox shell. The adaptations include:
    * change of the exit method to raise a
      :py:class:`ShellArgumentParserExitException`
      instead of invoking `sys.exit()`.
    * Two docorators :py:meth:`do_with_args` and `complete_with_args`
      intended to realize shell commands with argument parser.

    
    """

    def exit(self, status:int =0, message: str=None) -> None:
        """Overwrite the exit method to not exit the system but to
        raise an exception.

        Raises
        ------
        ShellArgumentParserExitException:
            This exception will be raised whenever exit is invoked.
        """
        if message:
            print("Parser:", message)
        raise ShellArgumentParserExitException(status)

    def do_with_args(self, function):
        """A decorator for do_... functions.
        This decorator will parse command line arguments and also
        catch exceptions raised by the command.
        """
        @wraps(function)
        def closure(shell: Cmd, inp: str):
            try:
                args = self.parse_args(inp.split())
                return function(shell, args)
            except ShellArgumentParserExitException as exception:
                print(f"tool: exit status {exception.status}")
            except Exception as exception:
                handle_exception(exception)
        return closure

    
    def _option_strings(self, text: str='') -> Iterable[str]:
        """Auxiliary function to provide a list of command line
        arguments, starting with a given substring.
        """
        # FIXME[hack]: we need an official way to obtain the option strings
        # - here we use the private list ._actions
        for action in self._actions:
            for option_string in action.option_strings:
                if not text:
                    yield option_string
                elif (text[0] == '-' and (len(text) < 2 or text[1] != '-')):
                    # a single '-': looking fo a short option
                    if (len(option_string) == 2 and
                        option_string.startswith(text)):
                        # only yield short options
                        yield option_string[1:]
                elif option_string.startswith(text):
                    # looking for long options (starting with double dash '--')
                    yield option_string[2:]

    def complete_with_args(self, function):
        """A decorator for complete_... methods.
        This decorator adds completion for command line arguments and also
        catch exceptions raised by the completion function.
        """
        
        def closure(shell: Cmd, text: str, line: str, begidx: int, endix: int):
            #print(f"complete: '{line[:begidx]}"
            #      f"[{line[begidx:endix]}]{line[endix:]}': text='{text}'")
            try:
                spaceidx = line.rfind(' ', 0, begidx) + 1
                if spaceidx < begidx and line[spaceidx] == '-':
                    # Trying to complete a command line argument
                    return list(self._option_strings(line[spaceidx: endix]))

                return function(shell, text, line, begidx, endix)
            except Exception as exception:
                handle_exception(exception)
        return closure

class ToolboxShell(Cmd):
    """An interactive shell to work with a :py:class:`Toolbox`.

    Notes
    -----
    The do_... methods implement the commands provided by the shell.
    If such a function returns a value interpreted as True, the command
    loop will stop - without return the return value is None,
    the loop will continue (this behaviour can be adapted by overwriting
    the postcmd hook).
    """

    prompt = '\033[1m' + 'dl-toolbox> ' + '\033[0m'
    intro = '\n'.join([_.strip() for _ in """
    Welcome to the dl-toolbox. Type \033[1mhelp\033[0m for help.
    """.splitlines()])

    def __init__(self, toolbox: Toolbox=None, **kwargs):
        super().__init__(**kwargs)
        self._toolbox = toolbox

    def emptyline(self):
        """React to empty lines.

        By default when an empty line is entered, the last command is
        repeated. We will disable this behaviour.
        """
        pass

    def do_error(self, inp):
        raise RuntimeError("Just a test error.")
    
    def do_quit(self, inp):
        """Quit the toolbox.
        """
        print("Quitting the toolbox.")
        if self._toolbox is None:
            self._toolbox.quit()
        return True
       
    do_exit = do_quit

    def do_bye(self, inp):
        """End the shell.
        """
        print("Good bye!")
        return True

    def do_EOF(self, inp):
        """End the shell.
        """
        print("EOF")
        return True

    def do_list(self, inp):
        """List resources available to the program.
        """
        print("Datasources:")
        for key in Datasource.register_keys():
            print(f" - {key}")
        print("Known subclasses of Datasource:")
        for key in Datasource.classes():
            print(f" - {key}")

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

    def _state_for_key(self, register, key) -> str:
        NORMAL = '\033[0m'
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        if not register.key_is_initialized(key):
            return YELLOW + 'uninitialized' + NORMAL
        obj = register[key]
        if obj.failed:
            return RED + 'failed' + NORMAL    
        elif obj.prepared:
            return GREEN + 'prepared' + NORMAL
        else:
            return BLUE + 'unprepared' + NORMAL
        
    _datasource_parser = ShellArgumentParser(description="Manage Datasources")
    _datasource_parser.add_argument('--list', help='List known datasources',
                              action='store_true')
    _datasource_parser.add_argument('--prepare', help='Prepare the datasource',
                                    action='store_true')
    _datasource_parser.add_argument('--unprepare', action='store_true',
                                    help='Unprepare the datasource')
    _datasource_parser.add_argument('datasources', help='Datasource to manage',
                                    metavar='DATASOURCE', type=str, nargs='*')

    @_datasource_parser.do_with_args
    def do_datasource(self, args):
        """Manage datasources.
        """
        if args.list:
            print("Known datasources are:")
            for key in Datasource.register_keys():
                print(f" - {key}  [{self._state_for_key(Datasource, key)}]")

            if not self._toolbox:
                print("datasource: error: no Toolbox")
                return

            print(f"Current toolbox datsource: {self._toolbox.datasource}")
            print(f"Other toolbox datsources:")
            for datasource in self._toolbox._datasources:
                print(f"  {datasource}")
            return

        for datasource_key in args.datasources:
            try:
                print(f"Initializing Datasource '{datasource_key}'")
                datasource = Datasource.register_initialize_key(datasource_key)
                if datasource.failed:
                    handle_exception(datasource.failure_exception)
                if args.prepare:
                    print(f"Preparing Datasource")
                    datasource.prepare()
                elif args.unprepare:
                    print(f"Unpreparing Datasource")
                    datasource.unprepare()
            except Exception as exception:
                print(f"Error: {exception}")
                handle_exception(exception)


    @_datasource_parser.complete_with_args
    def complete_datasource(self, text, line, begidx, endix):
        """Command line completion for the `datasource` command.
        """
        return [t for t in Datasource.register_keys() if t.startswith(text)]

    _network_parser = ShellArgumentParser(description="Manage Networks")
    _network_parser.add_argument('--list', help='List known networks',
                                 action='store_true')
    _network_parser.add_argument('--prepare', help='Prepare the network',
                                 action='store_true')
    _network_parser.add_argument('--unprepare', action='store_true',
                                 help='Unprepare the network')
    _network_parser.add_argument('networks', help='Network to manage',
                                 metavar='NETWORK', type=str, nargs='*')

    def _isinstance(self, obj, full_class_name) -> bool:
        module_name, class_name = full_class_name.rsplit('.', 1)
        if not module_name in sys.modules:
            return False
        return isinstance(obj, getattr(sys.modules[module_name], class_name,
                                       type(None)))
    
    @_network_parser.do_with_args
    def do_network(self, args: Namespace) -> bool:
        """Prepare a network.

        usage: network [NETWORK]

        Call `network --list` for a list of valid networks.
        """
        if args.list:
            print("Known networks are:")
            for key in Network.register_keys():
                print(f" - {key}  [{self._state_for_key(Network, key)}]")
            return

        for network_key in args.networks:
            try:
                print(f"Initializing Network '{network_key}'")
                network = Network.register_initialize_key(network_key)
                if self._isinstance(network, 'network.Classifier'):
                    print("- Classifier")
                if self._isinstance(network, 'network.Autoencoder'):
                    print("- Autoencoder")
                if self._isinstance(network, 'network.tensorflow.Network'):
                    print("- Tensorflow")
                if self._isinstance(network, 'network.keras.Network'):
                    print("- Keras")
                if args.prepare:
                    print(f"Preparing Network")
                    network.prepare()
                elif args.unprepare:
                    print(f"Unpreparing Network")
                    network.unprepare()
            except Exception as exception:
                print(f"Error: {exception}")
                handle_exception(exception)
            
    @_network_parser.complete_with_args
    def complete_network(self, text, line, begidx, endix):
        """Command line completion for the `network` command.
        """
        return [t for t in Network.register_keys() if t.startswith(text)]

    _tool_parser = ShellArgumentParser(description="Manage tools")
    _tool_parser.add_argument('--list', help='List known tools',
                              action='store_true')
    _tool_parser.add_argument('--prepare', help='Prepare the tool',
                              action='store_true')
    _tool_parser.add_argument('--unprepare', help='Unprepare the tool',
                              action='store_true')
    _tool_parser.add_argument('tools', help='Tools to manage',
                              metavar='TOOL', type=str, nargs='*')

    @_tool_parser.do_with_args
    def do_tool(self, args: Namespace) -> bool:
        """Prepare a tool.

        usage: tool [TOOL]

        Call `tool --list` for a list of valid tools.
        """
        if args.list:
            print("Known tools are:")
            for key in Tool.register_keys():
                print(f" - {key}  [{self._state_for_key(Tool, key)}]")
            return

        for tool_key in args.tools:
            try:
                print(f"Initializing Tool '{tool_key}'")
                tool = Tool.register_initialize_key(tool_key)
                if args.prepare:
                    print(f"Preparing Tool")
                    tool.prepare()
                elif args.unprepare:
                    print(f"Unpreparing Tool")
                    tool.unprepare()
            except Exception as exception:
                print(f"Error: {exception}")

    @_tool_parser.complete_with_args
    def complete_tool(self, text, line, begidx, endix):
        """Command line completion for the `tool` command.
        """
        return [t for t in Tool.register_keys() if t.startswith(text)]

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
        print(f"Fetched: {self._toolbox.input_data}")

    def do_gui(self, inp):
        """Start a graphical user interface.
        """

        if not self._toolbox:
            print("gui: error: no Toolbox")
            return

        if self._toolbox._gui:
            print("gui: error: GUI was already started.")
            return

        print("panel: warning: experimental stuff! "
              "You may observe strange behaviour ...")
        rc = self._toolbox.start_gui(threaded=True, focus=False)

    def do_gtk(self, inp):
        """Start the GTK+ based graphical user interface.
        """

        if not self._toolbox:
            print("gui: error: no Toolbox")
            return

        if self._toolbox._gui:
            print("gui: error: GUI was already started.")
            return
        
        print("panel: warning: experimental stuff! "
              "You may observe strange behaviour ...")
        rc = self._toolbox.start_gui(gui='gtkgui', threaded=True, focus=False)
        # FIXME[hack]
        #from gtkgui import create_gui
        #gui = create_gui()
        #gui.run()

    def do_panel(self, inp):
        """Open panel in graphical user interface.
        """

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

    def complete_panel(self, text, line, begidx, endix):
        if not self._toolbox or not self._toolbox.gui:
            return []  # gui was not started
        
        return [p for p in self._toolbox.gui.panels() if p.startswith(text)]

    def do_threads(self, inp):
        """List threads currently running.
        """
        import threading
        for thread in threading.enumerate():
            print(thread)

    def do_server(self, inp):
        """Start a HTTP server.

        The server allows (some limited) access to the Deep Learning
        Toolbox from a webbrowser.

        Available subcommands:

        info          provide status information on the server
        start         start the server
        stop          stop the server
        open          open a browser window pointing to the server
        """
        if not inp or inp == 'info' or inp == 'status':
            print(self._toolbox.server_status_message)
        elif inp == 'start':
            print(f"Starting the HTTP server")
            self._toolbox.server = True
            print(self._toolbox.server_status_message)
        elif inp == 'stop':
            print(f"Shutting down the HTTP server")
            self._toolbox.server = False
        elif inp == 'open':
            print(f"Opening server page ({self._toolbox.server_url}) "
                  "in web browser")
            self._toolbox.server_open()

    def do_sound(self, inp):
        """Play some demo sound.
        """       
        soundfile = None
        for directory in (os.environ.get('HOME', False),
                          os.environ.get('NET', False)):
            if not directory:
                continue
            soundfile = os.path.join(directory, 'projects', 'examples',
                                     'mime', 'audio', 'wav', 'le_tigre.wav')
            if os.path.isfile(soundfile):
                break
            soundfile = None
        if soundfile is None:
            print("error: no soundfile provided")
            sys.exit(1)
        print(f"Soundfile: {soundfile}")

        print("Creating reader")
        reader = SoundfileReader()

        print("Reading sound")
        sound = reader.read(soundfile)
            
        print("Creating player")
        player = SoundDevicePlayer()

        print("Playing sound")
        player.play(sound)

        print("Finished playing sound.")


if __name__ == '__main__':
    ToolboxShell().cmdloop()

    
