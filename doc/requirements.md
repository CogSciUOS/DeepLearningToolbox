# Requirements


## Minimal requirements


* python >= 3.6

* numpy


# Optional packages

## Tensor packages

### tensorflow

Tensorflow has undergone some substantial changes in the transition
from Tensorflow 1 to Tensorflow 2. As Tensorflow 2 offers a backwards
compatibility mode, in most situations it will be best to install
Tensorflow 2 and most examples should run. On the other hand,
if using Tensorflow 1, some newer code will not work.


### torch

When installing torch and a dedicated graphic card is available,
one should double check if it is used by the torch installation.




## Tools

### dlib

`dlib` is a machine learning library created by Davis King. It
provides several pretrained models for different tasks, including face
detection, landmarking and recognition. 

A recent version of `dlib`can be installed from PyPi
(this requires `cmake` for building a wheel and 
installation fails if `cmake` is not installed):
```sh
pip install dlib
```

The conda version seems somewhat outdated:
```sh
conda install -c menpo dlib
```

`dlib` provides the following tools:
* `hog`: a history of gradient based face detector.


# Networks


## Adversarial examples

### Cleverhans

```sh
pip install cleverhans
```

```sh
conda install -c conda-forge cleverhans
```


# Multimedia

## Images

## Videos

## Audio

### Package `sounddevice`

```sh
conda install -c conda-forge python-sounddevice
```

### Package `soundfile`

The package `soundfile` (formerly known as `pysoundfile`) is a wrapper
around `libsndfile` allowing to read and write sound files in Python.

```sh
pip install SoundFile
```


## Miscellaneous packages

### `setproctitle` (optional)

The package [setproctitle](https://github.com/dvarrazzo/py-setproctitle)
allows to get and set the process title (shown by the `ps` command).
Currently this is purely cosmetic and not installing `setproctitle`
will have no negative impact (all functionality will still be available).
Installation of `setproctitle` can be done by:

```sh
pip install setproctitle
```


# Datasets

