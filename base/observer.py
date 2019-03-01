from observer import Observable as BaseObservable, Observer as BaseObserver, BaseChange, change

class Observable(BaseObservable):
    """A :py:class:Observable object is intended to notify
    :py:class:Engine and user interfaces to be notified on changes.

    """

    # FIXME[problem]: arguments seem to be not accumulated over the
    # subclass hierarchy, i.e. if am.Config inherits from base.Conf
    # (which is an Observable) amd am.Config provides changes=..., but not
    # method=..., then the value for method is taken from here ("changed"),
    # not from base.Config
    def __init_subclass__(cls, changes=['changed'],
                          default='changed',
                          method='changed'):
        cls._change_method = method
        cls.Change = type(cls.__name__ + ".Change", (BaseChange,),
                          {'ATTRIBUTES': changes})
        cls._default_change = default
        def XChanged(self, observable:cls, info:cls.Change):
            raise NotImplementedError(f"{type(self).__name__} claims to be "
                                      f"{cls.__name__}.Observer "
                                      f"but does not implement {method}.")
        cls.Observer = type(cls.__name__ + ".Observer", (BaseObserver,),
                            {method: XChanged})

    def __init__(self):
        super().__init__(self.Change, self._change_method)



class TrainingObservable(Observable,
                         changes=['training_changed', 'epoch_changed',
                                  'batch_changed', 'metric_changed'],
                         default='training_changed',
                         method='trainingChanged'):

    def __init__(self):
        super().__init__()
        self._running = False
        self._epochs = None
        self._epoch = None
        self._batches = None
        self._batch = None
        self._loss = None
        self._accuracy = None
        self._validation_loss = None
        self._validation_accuracy = None

    @property
    def running(self):
        return self._running

    @property
    def epoch(self):
        return self._epoch

    @property
    def epochs(self):
        return self._epochs

    @property
    def batch(self):
        return self._batch

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batches(self):
        return self._batches

    @property
    def loss(self):
        return self._loss

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def validation_loss(self):
        return self._validation_loss

    @property
    def validation_accuracy(self):
        return self._validation_accuracy

