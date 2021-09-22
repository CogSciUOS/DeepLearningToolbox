

# Requirements

* Python version >= 3.8
* module `packaging`
* module `imageio` or some other module that provides `imread`
* module `frozendict`: for network
* module `numpy` with following functions:
  - `numpy.take_along_axis` (new in version 1.15.0)

Big modules (should be made optional!):
* module `matplotlib`:
* module `tensorflow`: with GPU support if suitable. Version 2.0 seems
  not to be supported yet!
  (in Tensorflow 2 we should use: `tf.compat.v1.GraphDef()`
  instead of `tf.GraphDef()`)
  ```python
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  ```
* module `PyQt5` (package `pyqt`):
* module `cv2` (package `opencv`):
* module `dlib`: dlib-19.19 (conda-forge)
* module `imutils`: tools/face/detector.py

# Installation with conda

```sh
conda create --name dl-toolbox python=3.6
source activate dl-toolbox
conda install packaging
conda install imageio
conda install -c conda-forge frozendict
conda install -c anaconda tensorflow-gpu=1.14
conda install -c anaconda pyqt
conda install -c conda-forge opencv
conda install -c conda-forge matplotlib
conda install -c conda-forge dlib
conda install -c conda-forge imutils
```

It may be possible to install some packages from other channels,
e.g. `matplotlib` should also be in the default channel. Check this!


The `menpo` channel provides Dlib (19.9-py35_0), which 
conflicts with many packages, e.g. python=3.6*
``` 
conda install -c menpo dlib
```

There is a dlib installation on conda-forge;
```sh
conda install  dlib
```


Incompatibilities:

* tensorflow 1.10.0 has requirement numpy<=1.14.5,>=1.13.3



To do
* end running threads (e.g. loops) gracefully
* turn off (unprepare) the webcam (or other data sources) if they
  are no longer used
