# Tools

## The three APIs

The Deep Learning Toolbox aims to provide access to a large set of
tools with a simple, yet flexible interface that allows for different
use cases.  Hence the `Tool` class provides three different interfaces.

### Functional API


### Data API


### The internal API


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

Furthermore, a worker provides means to queue data to process them
sequentially.



## Pre- and Postprocssing


### Preprocessing

Data preprocessing my include different kinds of operations:
* data normalization
* resizing (e.g., in case of images)
* internal representation (e.g. a framework specific data type, like
  a torch.Tensor or a tensorflow.Tensor.


Remarks:
* it is not clear to me yet, whether there is a preferred order in which
these preprocessing operations should be applied:
  - certain operations (e.g. normalization) are faster on smaller than
    on larger input (but do they loose accuracy?)
  - certain operations may be performed faster in a specific framework,
    while others may not be available
* there should also be a corresponding postprocessing operation
   - the postprocessing operation may be an inverse operation,
     in case the tool operates on the data, but it may also be
     applied on different data, e.g., in the case of a Detector


### Postprocessing


## Implementation

A tool implements pre- and postprocessing by specific (private)
methods:
* `_preprocess`: maps data to the internal format that can be passed
  to `_process`.
* `_postpropcess`: maps the result from the internal format to the
  input format. This method should also be supplied with the original
  input data so that it is able to derive the desired format.
  



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
