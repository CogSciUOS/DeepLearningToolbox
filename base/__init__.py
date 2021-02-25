"""base: a module providing base functionality for the deep
visualization toolbox.

This module should not depend on other modules from the toolbox.


The module supports the idea of observable objects. Such an object may
register observers and notify these observers upon certain changes.



A special type of observable objects is the :py:class:`BusyObservable`,
that can enter a busy state, meaning that the object is busy and certain
operations are blocked until the object has finished its job. When
calling busy methods, one may specify an additional parameter busy
to specify how to proceed in case that the object is busy

* 'block': wait until the object is no longer busy
* 'queue': FIXME[what]?
* 'exception': raise a BusyException

A :py:class:`BusyObservable` provides the property `busy`, to check
whether an object is busy and `busy_message` to get a textual
description of what the object is busy with.

A runnable object can mark methods by the `@asynchronous` decorator,
meaning that they may be executed asynchronously (in a separate
thread).  These are typically methods that may take some time to
execute and would block the operation of the main task, e.g., the
event loop of a user interface.  The decorator determines whether
execution should happen synchronously (blocking) or asynchronously
(non-blocking) according to the following rules:

* if the `asynchronous` flag is given, it determines the mode of operation
* otherwise the behaviour depends on whether the current thread is an
  event loop thread.

Notice the relation of the ideas of observable, business and
asynchronization:

 * Observation is a way to register callbacks to be
   invoked upon certain changes (usually the change of state of an
   object), either synchrously (immediately) in the same thread or
   asynchrously, potentially in another thread.

 * Business is only relevant in a parallel setting, when different
   threads/processes may try to access a resource simultanously (this may
   also include changing the internal state of some object).

 * Asynchronization is only possible in a parallel setting and is
   actually the main mechanism to realize parallelism in the toolbox.

A method should be marked as `@busy`, if it requires exclusive access
to some resource(s). It does not specify if this an expensive
operation or not. An expensive method should be marked as
`@run`. There may be busy methods that do not require their own taks
(but they may be invoked from some task) and there may be `@run`
methods that do not make an object busy.

Example: Datasource:
* synchronously obtaining an element: this may not change the state
  of the datasource and hence does not need to be marked as `@busy`.
  However, it may change the state of a Datasource (e.g. current frame
  in a video) or require exclusive access (e.g. reading an image from
  a webcam), in which case it should be marked as `@busy`
* asynchronously obtaining an element:


Asynchronous operations that deliver no result
----------------------------------------------

Examples:
* preparation/unpreparation of some resource

Asynchronous operations that deliver some result
------------------------------------------------

Asynchronous operations that deliver some result need some special
attention: We do not only have to inform the the caller (or more
generally the observers), that the operation was performed, but we
also have to provide the the result. There are different ways to
achieve this goal:

* include the result in the notification of the observers
* store the result in the object and provide some method to access it.

Examples:
* fetching data from a datasource

"""

from dltb.base.types import Identifiable
from dltb.base.observer import Observable, change
#from .fail import Failable, FailableObservable
#from .busy import BusyObservable, busy
#from .prepare import Preparable
from .config import Config
from .runner import Runner, AsyncRunner
from .controller import Controller, View, run
