"""Ensure that the Deep Learning Toolbox prerequisits are satisfied.

Prerequisits are hard requirements, without which the Deep Learning
Toolbox is not able to run.  These currently include the Python
version (at least Python 3.6), the availability of certain core types
(either from a recent standard library or by some compatibility
packages) as well as certain third-party modules.

The use of third-party modules in the core of the Deep Learning
Toolbox (`dltb.*` except `dltb.thirdparty`) should be reduced to an
absolute minimum to allow the Toolbox to be used in as many
environments as possible.  In order to achieve this goal, the
Deep Learning Toolbox should follow a flexible implementation scheme,
making third-party dependencies optional by either
(1) providing simple fallback implementations, or
(2) making the functionality optional,
that is, the Deep Learning Toolbox should also run without these
dependencies.  In such situations, the user may be informed that
addtional functionality would be available when these dependencies
are installed.

Currently the Deep Learning Toolbox has the following hard
third-party dependencies:
* Python >= 3.6
* typing_extensions (for Python <3.8)
* packaging
* numpy


Ideas for future improvements
-----------------------------

* Do not exit immediatly, but rather collect a list of all unsatisfied
  prerequisits and output them, as that would ease the task of setting
  up a suitable environment.
* Also include soft (not required) dependencies into that list.
* Do not exit but rather raise an Exception (e.g., ImportError or
  ModuleNotFoundError) so that code importing the Deep Learning Toolbox
  has a chance to react to the problem (e.g. not use the toolbox or
  install missing dependencies).
* There seems to be some old checking code in util/check.py
"""
# standard imports
import sys
import importlib

# This module should not make use of any thirdparty modules and it
# should also avoid importing modules from the Deep Learning Toolbox,
# as those may make use of unsupported features.


# Prerequisite 1: Python 3.6 (for f-strings)
#
# For older Python versions, one could opt for using the module
# `future-fstrings` (pip install future-fstrings).  However, this
# module requires to mark files with a specific coding hint, which
# is currently not done in the Deep Learning Toolbox, and there is
# probably no good reason to do so, as `future-fstrings` has been
# depcrecated by its author [1], as both, python 2 and python3.5 have
# officially reached end of life.
#
# [1] https://github.com/asottile-archive/future-fstrings
# [2] https://endoflife.date/python
if sys.version_info[0] < 3 or sys.version_info[1] < 6:
    print("Your python is to old: " + sys.version, file=sys.stderr)
    print("The Deep Learning Toolbox requires at least Python 3.6",
          file=sys.stderr)
    sys.exit(1)


# Prerequisite 2: Python 3.8 or the module typing_extensions
#
# See dltb(/typing.py)
if sys.version_info[1] < 8 and \
          importlib.util.find_spec('typing_extensions') is None:
    # Insufficient typing support
    print("Insufficient typing support. "
          "Consider installing the package 'typing_extensions' or "
          "updating your Python installation to 3.8 or above.",
          file=sys.stderr)


# Prerequisite 3: version comparison
#
# For comparing version strings, there are essentiall three options,
# which all have some their own drawback:
#  * version.parse() from the module 'packaging'. Packaging is a
#    third-party module and hence may not be available.  However, it
#    is used by setuptools and is conformant to the current PEP 440 [1]
#  * the LooseVersion and StrictVersion functions from the package
#    'distutils.version'. These are undocumented and conform to the now
#    superseeded PEP 386.
#  * manual parsing, which is more effort and also prone to errors.
#
# In the Deep Learning Toolbox, we decided to go for the 'packaging' module.
# However, in the future we may remove this prerequisite by implementing
# a more flexible solution.
#
# 'packaging' is currently used in the following files:
# - util/check.py
# - models/example_keras_vae_mnist.py
# - network/keras.py
# - dltb/thirdparty/keras/__init__.py
# - dltb/thirdparty/tensorflow/v2.py
# - dltb/thirdparty/tensorflow/v1.py
# - dltb/thirdparty/tensorflow/_postimport.py
# 'distutils'
# - util/check.py is currently used in the following files:
#
# [1] https://www.python.org/dev/peps/pep-0440/
# [2] http://www.python.org/dev/peps/pep-0386/
if importlib.util.find_spec('packaging') is None and \
       importlib.util.find_spec('distutils') is None:
    print("Module 'packaging' was not found. "
          "The Deep Learning Toolbox relies on 'packaging'. "
          "Please consider installing it.", file=sys.stderr)
    sys.exit(1)


# Prerequisite 4: Numpy
#
# The Deep Learning Toolbox currently makes use of Numpy as a central
# mechanism to store array.
#
# It could be possible to weaken this prerequisite by introducing an
# abstract `Array` class that could then be filled by different
# implementations (e.g. Tensorflow or Torch Tensors).  However, as
# these libraries have Numpy as hard requirement, there seems to be no
# real benefit in doing so.
if importlib.util.find_spec('numpy') is None:
    print("No numpy was found. "
          "The Deep Learning Toolbox relies on numpy. "
          "Please consider installing it.", file=sys.stderr)
    sys.exit(1)
