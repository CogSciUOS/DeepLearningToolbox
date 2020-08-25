import sys
import time
import importlib
import multiprocessing


# FIXME[experimental]: this is experimental code! Move it to some more
# suitable location.

class Process(multiprocessing.Process):
    """Create parallel processes and establish interprocess communication.

    A new background process can be prepared by obtaining a new
    instance of this class (`process = Process()`), and it can be
    started by calling `process.start()`

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._queue = multiprocessing.Queue()

    def set_title(self, title) -> None:
        try:
            proctitle = importlib.import_module('setproctitle')
            print("Old process title: ", proctitle.getproctitle())
            proctitle.setproctitle(title)
            print("New process title: ", proctitle.getproctitle())
        except ImportError:
            print("Changing process title is not supported, sorry!",
                  file=sys.stderr)

    def run(self):
        count = 0
        self.set_title(f"Deep Learning ToolBox: {self.name}")

        while True:
            if not self._queue.empty():
                message = self._queue.get()
                print(f"Process '{self.name}' received message '{message}'")
            print(f"Process '{self.name}' is active: {count}")
            count += 1
            time.sleep(1)

    def send(self, message: str) -> None:
        self._queue.put(message)
