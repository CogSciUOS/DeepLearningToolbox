

# Models

Models in the context of deep learning are usually deep networks (in
the Deep Learning ToolBox represented by the Network class).  However,
when talking abstractly about models, this is not necessary the case
and we want to allow for a more abstract view, allowing to specify
properties and methods of a specific type of model independent of its
actual nature. For this reason, the Deep Learning ToolBox introduces
the class `Model` and several subclasses realizing specific types of
models.

## General remarks

### Single vs. batch processing


* classify() -> automatic
* classify_single() -> 
* classify_batch() ->

## Classifier

A classifier is associated with a classification scheme, specifying
the number of classes and optionally some additional description, like
textual labels.

### Soft classifier

A soft classifier does not output one specific class but rather a
("probability") distribution over the classes. Of course, every soft
classifier can be turned into a hard classifier by just returning the
class with hightest confidence value.


# Networks

## Sequential network

A sequential network is composed of a sequence of layers.
