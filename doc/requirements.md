# Requirements


## Minimal requirements


* python >= 3.6

* numpy


# Optional packages

## Tensors

### tensorflow

### torch


## Tools

### dlib

`dlib` is a machine learning library by Davis King. Provides several
pretraiend models for different tasks, including face detection,
landmarking and recognition. It provides the following tools:
* `hog`: a history of gradient based face detector.

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

