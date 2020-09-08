# Deep Learning ToolBox Development Principles

* keep the implementation simple and clear: each component and method
  should realize a simple idea, that can described in simple terms and
  exhibited through a well-defined interface.

* separate different aspects: for example separate the core implementation
  of a network from the logic to invoke it asynchronously from a
  graphical user interface. The idea is that this should help
  to keep the implementation clear, allowing to focus on the actual
  concepts without becoming distracted from technicalities.

* be framework agnostic: provide an interface that can work with
  different frameworks and backends

## Method invocation

The Deep Learning ToolBox is aimed at being useable as a library for
implementing deep learning programs as well as a backend for a
graphical user interface.

One challenge in developing a graphical user interface (GUI) is to
maintain a smooth user experience. This is usually achieved by some
signaling mechanism, where user interaction trigger computations to be
started in the background, while in the foreground the event loop is
responsible for keeping the the graphical interface responsive. The
interface should reflect the state of the computation. This may mean
that it indicates that some component is busy and for long operations
to display its progress.

On the level of the programming interface of the Deep Learning ToolBox
this means, that the toolbox should provide mechanisms to start
operations in the background and to inform interested parties on the
progress and result of the computation.

As the Deep Learning ToolBox is intended to be framework agnostic, it
does not rely on an library specific signaling technique, but rather
provides its own interface together with code to map it to other
frameworks if required.


## Synchronous and asynchronous invocation

A central difference between synchronous and asynchronous computation
is, that the first can usually be realized in a stateless fashion,
returning the result via the function call stack to the caller.  In a
synchronous computation, the result has to be stored in order to be
accessible to the initiator the computation (and maybe other
interested parties).

In the toolbox we account for this by separating the computation and
the asynchronous invocation into different classes. The asynchronous
invocation class will hold a reference to a computation class.
Upon invocation, it will use that class for computation and store
the result in a local attribute and notify observers once the computation
is complete. The invocation may be synchronous (in library mode) or
asynchronous (in GUI mode).

 * Principle: all asynchronous methods should also be callable in a
   synchronously (blocking)

When called synchronously:
* no Thread should be created
* the result should still be stored (if there is some result)
* observers should still be informed

The decision wether to run synchronously or asynchronously is based on:
* an explicit paramter (`run: bool=None`)
* some default mechanism (checking the nature of the current thread)

Methods eligible to asynchronous (threaded) execution should be
decorated with the `@run` decorator. Typical examples are:

* the `prepare()`and `unprepare()` methods
* methods fetching data and storing it
* methods performing some computation and storing the result
* methods for training a network
* methods for playing a sound or a video

Criteria to decide if a method should be decorated with `@run`:
* if the method performs a complex operation that may require some time
  (and block a user interface), then it is a candidate for asynchronous
  (threaded) execution.

* A `@run` method must not return any result. If it does, it should
  be redesigned by storing the result in some object property.
* A `@run` method must notify observers. Notification has at least
  be done when the operation finishes, but may in addition happen
  at the beginning or when some progress is achieved. It is sufficient
  to combine it with the `@busy` decorator.

The `@run` decorator can be combined with the `@busy` decorator.
In this case the order is important: it should be `@run @busy`, as
the busy notifications should send when the thread starts and stops,
not when the wraper ends.



Examples:
* Datasource vs. Datafetcher: there may be multiple datafetchers 
  per datasource, 
  FIXME[todo]: a fetcher may also provide loop functionality 
* Network vs. ActivationTool: again ther may be multiple tools per network


An asynchronous invocation class may be busy, meaning that it is
currently perfoming a computation. It may then handle successive calls
in different ways:
* reject: don't allow any further computation before the current
  computation has finished. Throw an Exception.
* queue: put further computation requests into a queue and process
  them once the current computation is finished
* last: just queue the most recent computation request and process
  it once the current computation is finished
* replace: abort the current computation (as soon as possible) and
  start instead the new computation request


An asynchronous invocation class may provide an abort method. Such
a method may abort the current operation, dropping all (intermediate)
results (the result field of the invocation class will not be changed
and no value_changed notification will send, just a busy changed
message may be sent).





### Asynchronous (blocking, non-threaded)

```python
tool.do_computation()
```

```python
def do_computation():
    self.notify_start()  # notify observers
    
    
    
    self.notify_end()  # notify observers


### Synchronous (Threaded, non-blocking)


```python
tool.start_computation()

tool.stop_computation()
```

```python
def start_computation():
    self.thread = start_computation_in_thread(self.wrap_computation)

def wrap_computation()
    self.thread = start_computation_in_thread()
    self.result = 
    
    self.notify_observers('state')
```


## Locking and business

Components may become busy by performing an operation, meaning they
are at that time not available to perform other operations (or the
same operation on other data). Examples:

* a network that is performing training. The network is not available
  to perform infernce tasks at the same time.
  
* a network performing von inference task is maybe not available to
  do other inference tasks at the same time, as this may require to
  swap data between CPU and GPU memory, which is prohibitively
  expensive. However, in other situations it may be allowed to
  do multiple inference steps in parallel.

* a webcam that is running a streaming loop, that is grabbing one
  image afte the other. Such a webcam can not be used to make a
  snapshot.
  
* a music player that is outputing audio data to a sound device
  may not at the same time output a second audio.

* an object that is currently in preparation (loading resources)
  can not process any data.

In all these situations, the component should signal its business by
setting the busy flag. It should inform observers that the busy state
changed (on starting and on finishing the job). It may also lock
access to methods that are not available, either throwing an exception,
or blocking the calling thread until the component is available again.




## Notifications



## Stateless and stateful objects

A stateless object (or better an object with frozen state) can be
called in parallel. On the other hand, a stateful object (an object
that changes) has to be treated with care in parallel environments.

* stateless / no threads / synchronous

* stateful / threaded / asynchronous -> sutiable for GUI

It may make sense to split objects into a stateless and a stateful
part:

* Datasource: the actual stateless Datasource allowing to load data
  in a synchronous way, and a Datafetcher, which is based on a
  Datasource and can fetch data asynchronously, storing the result
  as part of its state, notifying observers when done, etc.
  
* Network: the actual network can compute activations synchronously,
  while the activation tool can use a network to compute activations
  and asynchronously store results and notify observers. However,
  a trainable network obviously has some state (the connection weights),
  though this seems to be somewhat different.
  
For some components, their nature is not so clear to me:

* The toolbox definitively hold some state: the current input data,
  the set of datasources, networks, tools, etc.

* A Trainer has some state (epach, batch) - it should probably
  notify observers when these change.


# Initialization and preparation




# Open points:

* integrate batch data: a lot of tools and graphical components do not
  know how to deal with batch data correctly.

* states: the states of stateful object should be designed in a clearer way
  (including notifications that signal the change of state):
  - busy: the object is currently busy doing some task
  - prepared: the object was prepared. if False, the object
    is unprepared and can be prepared by calling prepare()
  - failed: the object is in a failed state and should not be used.
    calling clean may reset it into a healthy state
  - ready: the object is ready and can be used:
      - if Preparable, the object has to be prepared
      - if Busy, the object must not be busy
      - if Failable, the object must not be in the failed state
  * we have to distinguish objects that are busy and cannot do another
    task (e.g. a webcam that is streaming, or a network that is
    using/blocking the GPU, an object that is currently preparing -
    it should not start a second preparation ...) and those that are
    busy but can be called again (e.g. a processor that is queueing requests).
  * threading: we may start a background thread and feed this thread
    from a queue (like a Processor), or we may just start a new
    background thread for every call. In the first case we need a
    mechanism to check if a thread is currently running (including
    some synchronization logic ...)

  Some specific questions:
  - looping (in Datafetcher): maybe busy + some extra flag?
  - what is the state of a tool.Processor processing data.
    it should be busy (as it can not immediately react to new data)
    but it allows to queue new data (hence it is in some sense ready)
    maybe some extra state 'processing'? still the underlying tool
    can be busy, meaning the processor has to wait before it can
    apply that tool.

* running background tasks: there are currently multiple run/runner/async
  implementations. My goal would be one or at most two decorators:
  - run: run a task in the background
  - busy: mark object as busy, may (but doesn't have to) run in its
    own thread

  Some quesions have to be considered: may a busy object call other
  methods protected by @busy? probably yes (example: a looping 
  datafetch calls the @busy fetch method to obtain the next data).
  - should we provide names for running background threads?

* subprocessing:
  for some computations it would be good to run them in parallel,
  that is in their own thread or even in another process.


* lazy import:
  in packages `__init__.py` files, we should not
  
  
* FIXMES:
   30    datasource/
   71    network/
   66    toolbox/
   70    tools/
   36    dltb/
   55    base/
   17    util/
    3    demos/
    2    gui/  -> move to dltb.toolbox.gui
    0    dl-toolbox.py
  271    qtgui/
   
  EXPERIMENTAL:
    1    downloader.py
         maximize_activation.py

   OLD:
         controller/
         model/

# Subpackages of the Deep Learning ToolBox

## base

Abstract datastructures to be used by other classes of the toolbox.
Should nor rely on/import any other subpackage directly.

## util

Collection of convenience functions, different motivations:
* simple functional API, for simply accessing the class based API
  of the Toolbox
* functions for installing software and data

## datasource

Definition of the abstract `Datasource` class and several
implementations, not (directly) using third party modules.

## network

Definition of the abstract `Network` class and several
implementations, using different third party modules.


## tools

Definition of the abstract `Tool` class and specialized subclasses
and several implementations, using different third party modules.


## toolbox

The Toolbox makes use of the abstract intefaces defined in
base, datasource, network, and tools, as well as util.

The central `Toolbox` class provides a hub for requesting data, and
tools. It also coordinates starting the different possible interfaces.


### gui

Abstract definition of a graphical user interface framework as a base
for real implementations of such a interface.

### shell

A shell allowing to interactively query the toolbox.

### server

A webserver allowing to query the toolbox from a webbrowser.


## thirdparty

Implementations of the abstract classes from base using thirdparty
libraries.


## Individual files 

### argparse.py

### config.py

### directories.py
