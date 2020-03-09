
# FIXME[problem]: circular import:
#   BaseController -> AsyncRunner -> Observable -> BaseController
# from .asyncrunner import AsyncRunner


class BaseController:
    """Base class for all kinds of controllers.

    Attributes
    ----------
    _datasource : Datasource
        The current :py:class:`datasource.Datasource`
    """
    def __init__(self, runner: 'AsyncRunner'=None):
        self._runner = runner

    def set_runner(self, runner: 'AsyncRunner') -> None:
        """Set that the :py:class:`AsyncRunner` for this Controller.

        Parameters
        ----------
        runner: 'AsyncRunner'
            The runner to be used for asynchronous execution. If `None`,
            no asynchronous operations will be performed by this
            BaseController.
        """
        self._runner = runner

    # FIXME[design]: can there be multiple Observables that are
    # controlled by one Controller?
    def get_observable(self) -> 'observer.Observable':
        """Get the Observable for the Controller.  This will be the
        object, that sends out notification in response to commands
        issued by this controller. Everyone interested in such
        notifications should register to this Observable:

        Result
        ------
        model: Observable
            The model controlled by this ActivationsController.
        """
        raise Error("No default Observable available!")
