# Tools


FIXME[bug]: due to refactoring, the implementation is currently in an
insane (not operational) state: there seems to be some redundancy between
`Tool.apply` and `Detector._process_data`. Currently `apply` is
called from:
* `dltb/tool/worker.py`
* `dltb/tool/detector.py`
* `dltb/tool/activation.py`
* `dltb/tool/tool.py`: (`IterativeTool`)
* `demos/dl-activation.py`

The `process_data` family is used similar places. `_preprocess_data`:
* `dltb/tool/detector.py` (seems not to be called)
* `tools/activation/engine.py` (seems outdated ...)
the central `_process_data` (and `_process_batch`):
* `dltb/tool/tool.py`:
* `dltb/tool/detector.py`:
and finally `_postprocess_data`:
* `dltb/tool/detector.py`

This second set of functions (`_...process_data`) seems to be outdated
and should be removed!


## The three APIs

The Deep Learning Toolbox aims to provide access to a large set of
tools with a simple, yet flexible interface that allows for different
use cases.  Hence the `Tool` class provides three different interfaces.

### Functional API

This is the standard API of the tool.  It takes arguments in any of
the supported Deep Learning Toolbox types and returns the result also
in one of these formats.

```python
tool(datalike, arguments, result=(...)) -> Result
```

Preprocess `datalike` to obtain suitable internal representation of the
data and attributes (by calling `_do_preprocess`).

Then process the data in its internal representation (by calling `_process`).

Finally postprocess data to get the desired result (by calling
`_do_postprocess`).



### The internal API

The flag `internal` signals, that `value` and `arguments` are given
in an internal (preprocessed) format, ready to be passed to the internal
implemenation of the tool.
```python
tool(value, arguments, internal=True) -> internal_result
```
The internal interface is intended to directly interact with the
underlying implementation (which is usually only accessible via
a private API).





### Data API

The Data API allows to apply the `Tool` to a `Data` object.  Results are
stored as attributes of that data object.  The Data API is intended for
asynchronous processing.

```python
tool.apply(data, arguments, result=(...))
```

The function `apply` basically does the following steps:
* obtain some values by calling `self.__call__(data, result, ...)`
* store these values as data attributes (according to `result`)


The `Data` API is for example used by the `Worker`. An example
can be found at the "Face Panel".





## The result argument

The result argument allows to specify what result should be returned.
Each tool provides a list of potential results and the result argument
can name the results the caller is interested in.
* the result is returned as a singleton or a tuple according to the
  specification in the result argument.
* in case of `Data` processing, the results are stored as data atttributes
* each tool has a default result value, which will be used if no result
  argument is specified
* the prefix `_` is used for internal values.


## Internal state

The `Tool` is a `Stateful` object:
* it can be prepared and unprepared
* it may be busy (if it does not allow for parallelization)
* it may be in a failed state (when prepartion fails)


## Synchronous invocation

The application of tools is implemented in a functional, stateless
fashion. While the `Tool` itself is as `Stateful` object, allowing for
preparation or failure, the actual application of the `Tool` should
not change its state (it may block, if it does not allow for parallel
application).


### Calling the `Data` API synchronously
When using the `Data` API


### The `Worker`

The Worker can apply a tool, either asynchronously or synchronously.
The results of an application are stored as attributes. There
are three modes of a worker:
* direct: data are provided without preprocessing in one of several
  formats that the toolbox can process and the results are returned
* internal: data is provided in internal (preprocessed) format and
  results are stored in the internal formal (returned by the underlying
  implementation) as well.
* data: data is provided as a `Data` object, which is stored as a
`data` attribute of the worker.

Furthermore, a worker provides means to queue data to work on them
sequentially.



## Pre- and Postprocssing


### Preprocessing

Data preprocessing my include different kinds of operations:
* data normalization
* resizing (e.g., in case of images)
* internalization: convert data into an internal representation
  (e.g. a framework specific data type, like a torch.Tensor or
  a tensorflow.Tensor). This may include moving the data to the
  desired computation device.


Remarks:
* it is not clear to me yet, whether there is a preferred order in which
these preprocessing operations should be applied:
  - certain operations (e.g. normalization) are faster on smaller than
    on larger input, so shrinking but not growing an image should be
    done before (but do we loose accuracy?)
  - certain operations may be performed faster in a specific computational
    framework (like a GPU), while other operation may not be available
    for that device.
* there should also be a corresponding postprocessing operation
   - the postprocessing operation may be an inverse operation,
     in case the tool operates on the data, but it may also be
     applied on different data, e.g., in the case of a Detector


### Postprocessing



## External and internal data

All tools of the core of the Toolbox are defined in terms of general
(abstract) datatypes.  The motivation is to make the description of a
tool independent of a specific implementation.

When using a `Tool`, the user should not assume any specific
implementation or specific data representation.  It should be possible
to call the tool with different variants of a datatype (e.g. numpy
array, torch or tensorflow Tensor, etc.)  and the tool should
automatically convert this into the desired format.  All of this
should be as transparent as possible to both, the user applying the
`Tool` and the developer implementing the `Tool`.


# Implementing a new `Tool`

A `Tool` implements pre- and postprocessing by specific (private)
methods:
* `_preprocess`: creates a `Data` object and fills it with values that
  may be used for processing and postprocessing.
* `_process`: apply the tool to the preprocessed data.
* `_postprocess`: maps the result from the internal format to the
  output format. This method gets the same `Data` object,
  to which the desired values should be added. The object also
  provides additional information that can be used for doing
  the postprocessing.

Remark: the `Data` object can be seen as a fresh namespace, allowing
to store values. It is not the `Data` object that is provided when
invoking the tool, but a temporary object, that is just created for
passing data betwen functions. (could we replace it by a simple dict?)



```python

class MyTool(Tool):

    external_result: Tuple[str] = ('my_result', 'duration')
    internal_arguments: Tuple[str] = ('_preprocessed', )
    internal_result: Tuple[str] = ('_result, )

    def _preprocess(data: Datalike, arg1: Any = None, **kwargs) -> Data:
        # return preprocessed data and arguments that can
        # be passed to process()
        data = super()._preprocess(*args, **kwargs)
        data.add_attribute('mean', data.data.mean())
        return data
        
    def _process(self, preprocessed) -> Any:
        # the actual processing
        # return processed data in internal format
        return preprocessed**2
        
    def _postprocess(data: Data, name: str) -> None:
        if name == 'my_result':
            data.add_attribute(name, data.mean)
        else:
            super()._postprocess(data, name)
```

# Iterative Tools


## Tool API

An iterative tool may be run like every tool, just that the
application may take more time as it may require several iteration
steps to finish.


## Iterative API

In addition to the standard `Tool` API, an iterative tool provides an
iterative API, that allows to perform individual iteration steps.



```python
result = tool.step(data, arguments)
```


```python
for intermediate in tool.steps(initial, arguments):
     do_something(intermediate)
```



Using a `Data` object that can be observed. The data object will be
updated (and Observers will be notified) after every step.
```python
tool.process_data(data, arguments)
```


Using a processor: the processor will run the tool and hold the
current state as attributes, which can be read out by interested parties.
```python
processor = IterativeProcessor(tool)
processor.process(initial, arguments)
```

## Open issues


### Specification of results

* a step may not only produce an intermediate result, but also some
additional information an observer may be interested in:
- step number in the iteration
- loss value(s)
- timing information

Idea: the desired result(s) can be specified by an (optional) argument:
result: Union[str, Tuple[str]]. These strings name the values to return.
Valid names have to be specified by the individual tools. A default value
may be set be each tool, which may be overwritten upon construction.


### Invocation of internal implementation

All the tools of the Toolbox should be specified using general Toolbox
data types. However, the actual implementation will often use some
third-party module with specific data formats.  The Toolbox API will
mostly act as an wraper around these internal functions. It may be
interesting for a caller to call the underlying functions directly
(either for performance reasons, or to study the internal data
returned by the underlying implementation. For this reason, all tool
functions should provide a corresponding `..._internal()` method,
that operates on the internal data. And each tool should provide
functions `preprocess()`, `preprocess_arguments()`,  and 
`postprocess()` method. The actual tool will essentially do

```python
internal = self.preprocess(*args)
result = self.apply_internal(*internal)
return self.postprocess(result, *args)
```

* the internal method should be public: under certain circumstances
it may be interesting to call it directly.




# List of tools

## Simple tools

### Classifier



### Detector

#### Face Detector


### Recognizer


#### Face Recognizer


## Itererative tools

### Activation maximization

The activation maximization operatates on input data, that are
(iteratively) transformed to maximize the activation of a unit in a
network. Hence additional parameters are the network and the
identification of layer and unit.


### Adversarial attack

A (iterative) adversarial attack operates on imput data, that are
(iteratively) transformed to increase some loss value. It requires a
model and a loss function.

This approach seems similar to the activation maximization.


### Style transfer

Style transfer iteratively transforms input data to match a given
style. Additional parameters are style, model and loss function.


### Trainer

Training (iteratively) trains a model, based on a dataset and a loss
function. Iterative values are the model and a loss value.
