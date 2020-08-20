import multiprocessing
import time


class Process(multiprocessing.Process):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._queue = multiprocessing.Queue()

    def run(self):
        count = 0

        while True:
            if not self._queue.empty():
                message = self._queue.get()
                print(f"Process '{self.name}' received message '{message}'")
            print(f"Process '{self.name}' is active: {count}")
            count += 1
            time.sleep(1)

    def send(self, message: str) -> None:
        self._queue.put(message)
