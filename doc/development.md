# Deep Learning ToolBox Development Principles


* separate different aspects: for example separate the core implementation
  of a network from the logic to invoke it asynchronously from a
  graphical user interface. The idea is that this should help
  to keep the implementation clear, allowing to focus on the actual
  concepts without becoming distracted from technicalities.

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

Examples:
* Datasource vs. Datafetcher
* Network vs. ActivationTool


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
