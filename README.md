![travis-ci](https://travis-ci.org/JarnoRFB/qtpyvis.svg?branch=master)

PyQt5 visualization of Deep Neural Networks (Development stage)
===============================================================
One can use this code to visualize DNNs (activations, filters, network structure).

Main goal of the toolbox is to visualize networks that solve image classifications tasks. Current version assumes to be given a model trained and saved using [Keras: The Python Deep Learning library](https://keras.io/ "Keras's Homepage")
Development of this toolbox was motivated by the paper:

 - Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson. [Understanding neural networks through deep visualization](http://arxiv.org/abs/1506.06579/ "Computer Science > Computer Vision and Pattern Recognition") Presented at the Deep Learning Workshop, International Conference on Machine Learning (ICML), 2015.


Functionality
=============

Example of a network visualization that classifies different shapes
![screenshot](doc/source/_static/example.png)

The main window consists of activation maps for selected layer, the input and the structure of a network where the layers of interest can be chosen

Installation
============

Installation process is just installing packages listed in the requirements.txt

- python virtualenv:

  1) Install your virtualenv following [The Hitchhiker’s Guide to Python](http://docs.python-guide.org/en/latest/dev/virtualenvs/ "Virtual Environments"), check that python version >=3.6 is set as primary interpreter

  2) After activating the enviroment, run the command to install necessary libraries:


        $pip install -r requirements.txt



Support for PyTorch
-------------------

Install PyTorch.

   `$ conda install pytorch torchvision cuda80 -c soumith`

Then run the PyTorch MNIST example, tpye

   `$ python main.py --framework=torch`


Testing
=======
Run `$ pytest` to execute all tests.

Documentation
=======

Building
--------
In order to build the Sphinx documentation, `cd` into the `doc` directory and
run `make html`. Then open `build/html/index.html`. __Note__: Sphinx executes
all found modules, which can exhibit side-effects. For instance, loading
TensorFlow takes a lot of time, and so building the documentation does as well.

Adding files
--------
When adding a new python file, run ``sphinx-apidoc <folder> -o doc/source`` to
generate an `.rst` file indexing the modules. When the apidoc tool finds modules
it has already created indexes for, it will complain that those already exist.
It may be neccessary to delete them and regenerate them if you add a submodule,
I'm not currently sure.
