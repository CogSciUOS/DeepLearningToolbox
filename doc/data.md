

# Data

Data is a subclass of Observable (Changeable)
* attribute_added
* attribute_changed

Data - Datalike
Image - Imagelike
Sound - Soundlike




# Datasource

* Datasource is `Observable`, Stateful (prepared/unprepared, busy, failed)

## The get API

## The context manager API



## Availabe Datasource

* Iterable: __iter__
* Indexed: __len__, __getitem__
* Life: awake, kill

* Imagesource
* Soundsource

* Directory: Indexed
* Array: Indexed
* Webcam: Iterable, Life
* Video: Indexed, Life
* Random: Life



# Datafetcher

* A `Datafetcher` can fetch from a `Datasource`
* A `Datafetcher` is `Observable`, it can notify observers once
  data is fetched.


### The API

* `fetch(...)`
* `loop(...)`


# Third-party datasources

## Keras

Keras provides the `keras.datasets` module (in the TensorFlow version
of Keras this is `tensorflow.keras.datasets`), which provide a few toy
datasets: MNIST, Fashion-MNIST, CIFAR10, CIFAR100 (image
classification) IMDB, Reuters (text classification), Boston house
prices (regression).

* Keras documentation: [Datasets](https://keras.io/api/datasets/)

## TensorFlow

TensorFlow offers multiple tools for data loading and data
preprocessing.

### TensorFlow `Dataset`


* TensorFlow: [Introducing TensorFlow Datasets](https://blog.tensorflow.org/2019/02/introducing-tensorflow-datasets.html)
* TensorFlow: [Writing custom datasets](https://www.tensorflow.org/datasets/add_dataset)
* TensorFlow documentation: 
  [tf.data.DatasetSpec](https://www.tensorflow.org/api_docs/python/tf/data/DatasetSpec)

### The package `tensorflow_dataset`

The package `tensorflow_dataset` provides access to different common
datasets.  The datasets will be provided as instances of
`tf.data.Dataset`.

The module `tensorflow_dataset` is not contained in the tensorflow
core packages but has to be installed separately. 
This can be done with Pip:
```sh
pip install tensorflow-datasets
```
or when using Conda:
```sh
conda install -c anaconda tensorflow-datasets
```

The package is usually imported as `tfds`.  It can be used as follows:
```python
import tensorflow_datasets as tfds
mnist_data = tfds.load("mnist")
mnist_train, mnist_test = mnist_data["train"], mnist_data["test"]
assert isinstance(mnist_train, tf.data.Dataset)
```

Some datasets will be downloaded automatically, while other datasets
require manual download (due to license restrictions).

* TensorFlow: [TensorFlow Datasets: a collection of ready-to-use datasets](https://www.tensorflow.org/datasets)


## Torch

Torch provides the `torch.utils.data.Dataset` module.  These
essentially implement `Sequence` interface, that is index access (via
the `__getitem__` method) and the `Sized` interface (the `__len__`
method).

### torch.utils.data: The dataloader

* Torch: [torch.utils.data](https://pytorch.org/docs/stable/data.html)

### Torchvision.dataset

* Torch: [torchvision.datasets](https://pytorch.org/vision/stable/datasets.html)

## Caffee

* [How to properly set up Imagenet Dataset for training in Caffe](https://github.com/arundasan91/Deep-Learning-with-Caffe/blob/master/Imagenet/How-to-properly-set-up-Imagenet-Dataset.md)
