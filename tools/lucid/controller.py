from base import Controller as BaseController
from network import Network
from tools.lucid import Engine


class Controller(BaseController):
    """Controller for :py:class:`tools.lucid.Engine`."""

    def __init__(self, engine: Engine, **kwargs) -> None:
        """
        Parameters
        ----------
        model: Model
        """
        super().__init__(**kwargs)
        self._engine = engine

    @property
    def engine(self) -> Engine:
        """Get the lucid activation maximization Engine for this
        Controller.  This is that actually performs the
        maximization process and that sends out notification in
        response to commands issued by this Controller. Everyone
        interested in such notifications should register to this
        Observable.

        Result
        ------
        engine: Engine
            The lucid engine controlled by this Controller.
        """
        return self._engine

    # FIXME[design]: this is only implemented to satisfy
    # Observer::set_controller()
    def get_observable(self) -> Engine:
        return self._engine


    def onMaximize(self):
        """Run the activation maximization process.

        """
        print("!!! run")
        self._runner.runTask(self._engine.start)


    def onMaximizeMulti(self):
        """Run the activation maximization process.
        """
        print("!!! run multi")
        self._runner.runTask(self._engine.start_multi)


    def onStop(self):
        """Run the activation maximization process.

        """
        print("!!! stop")
        self._engine.stop()

    # FIXME[design]: the following methods should actually go into the
    # Controller
    def onModelSelected(self, name):
        self._runner.runTask(self._engine.load_model, name)
