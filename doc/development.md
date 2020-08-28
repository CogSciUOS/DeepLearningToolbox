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


