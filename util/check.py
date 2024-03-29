import sys
import logging

# Begin checking for modules

# FIXME[todo]: check the packages that we actually need - This is not
# a good place to do here, and we need a more general concept to do
# this in a consistent way.

# Python
#
# We need at least python 3.6, as we make (heavy) use of the following
# features:
#  - PEP 498: Formatted string literals
#    [https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep498]
#  - PEP 526: Syntax for variable annotations
#    [https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep526]
if sys.version_info < (3, 6):
    logging.fatal("You need at least python 3.6, but you are running python "
                  + ".".join([str(_) for _ in sys.version_info]))
    sys.exit(1)

try:
    # Version checking.
    # We use the 'packaging.version' module, which is a popular
    # third-party module (e.g. used by setuptools) and which
    # is conformant to PEP 404.
    # An alternative would be to use the nowadays outdated
    # 'distutils.version' module, which is build in, but undocumented
    # and conformant only to the superseeded PEP 386.
    import packaging.version

    # new versions of tensorflow require recent versions of protobuf:
    #  - I experienced problems with tensorflow 1.12.0 with protobuf 3.6.1:
    #    tensorflow/python/keras/backend/__init__.py", line 22,
    #      from tensorflow.python.keras._impl.keras.backend import abs
    #    -> ImportError: cannot import name 'abs'
    #    However reinstalling both packages resolved the problem ...
    #    maybe not a version issue?

except ModuleNotFoundError as e:
    # FIXME[hack]: better strategy to inform on missing modules
    explanation = ("The module can be installed by typing: "
                   "conda install packaging")
    logging.fatal(str(e) + ". " + explanation)
    sys.exit(1)


# FIXME[hack]: this should be integrated into some more general check
# mechanism. We do it here explicitly, as 'frozendict' is required
# by (some) networks - but probably the cleaner way is to just
# disable these networks ...
# FIXME[update]: frozedict seems unmaintained and has been removed from
# the toolbox - mach sure that all traces have been deleted and then
# remove it ...
#from .importer import importable
#if importable('frozendict'):
#try:
#    import frozendict
#except ModuleNotFoundError as e:
#    explanation = "The module can be installed by typing: conda install -c conda-forge frozendict"
#    logging.fatal(str(e) + ". " + explanation)
#    sys.exit(1)


try:
    import numpy
except ModuleNotFoundError as e:
    explanation = "The module can be installed by typing: conda install numpy"
    logging.fatal(str(e) + ". " + explanation)
    sys.exit(1)


modules = {
    'packaging.version': {
        'level': 'critical',
        'explanation': 'needed for version checking',
        'conda': 'conda install packaging'
        }
#    'frozendict': {
#        'level': 'recommended',
#        'explanation': 'needed for some neural network models',
#        'conda': 'conda install -c conda-forge frozendict'
#        }
    }

# End checking for modules
