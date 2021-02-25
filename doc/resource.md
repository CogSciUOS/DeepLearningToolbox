# Resources

Several tools may use or even require specific resources to
operate. This may be:
* specific hardware, like a webcam or a GPU
* specific software, like a library or software package
* specific data, like a dataset or a pretrained model

Some of these resources may be hard requirements (a specific model may
not run without the TensorFlow library, the Webcam datasource will not
operate without a webcam), while others may be optional and just
improve the functionality (using a GPU may speed up computation,
having a specific library installed may allow to store images in more
formats). Some resources may be installable (python modules can be
installed with pip or conda, data can be downloaded), while others
need physical actions by the user (plug or grant access to a webcam).

## Checking for resources

To allow for a smooth user experience, the availability of resource
should be made transparent.  Components of the toolbox should check
if required resources are available before using them. Resources
are typically required at different points in time:

1. import: upon import, the code of a module is loaded and automatically
executed by the python interpreter. Typically, this are imports of
third-party modules, but it could also be other resources that are loaded
by top level code.
2. construction: class instantiation and intialization are usually the
places where resources needed by an object are acquired.
3. operation: when a object of function performs an operation.

However, the Deep Learning Toolbox tries to unify this process, by
establishing the following guidelines:
* in top-level module code, no resources should be required (as this may
  lead to exceptions upon import, which are hard to handle gracefully).
* object instantiation (`__new__`) and initialization (`__init__`)
  method should not require resources neither, as this may result in
  object construction to fail (which should be avoided in the Toolbox)
* during operation all required resources should already be acquired.
  Under normal circumstances, an operation should not fail due missing
  resources.

The main idea is to treat all resource related code in the preparation
process, which has to be invoked after object construction but before
using the object for actual operations. 


# Installation

A resource is considered installable, if it can be made available by
an installation process.  Installable resources should implement the
`Installable` interface, allowing to check if the resource is
installed and offering methods for installation and deinstallation.


# Requirements

Components of the Deep Learning Toolbox should make explicit what
resources they require.

A `ResourceUser` can register resources that are required either by
all instances of a class (class requrirements) or by one specific
instance (instance requirements).

The `install_requirements` method may be used to install missing
requirements.


# Preparation

Upon preparation, a component will check if all required


```
                 Resource
                 /      \
                /        \
        Installable      Hardware
         /      \
        /        \
    Package    Downloadable
```
