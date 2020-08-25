"""Central initialization file of the Deep Learning ToolBox (dltb).
"""

# The config module is the central interface for configuring the
# Deep Learning ToolBox. It is imported as one of the first modules,
# as it allows to influence the following initionalization process.
from .config import config

# Importing thirdparty will patch the machinery in order to
# adapt the import of some third-party libraries (like keras).
# This should be done before those libraries are imported for the
# first time.
from . import thirdparty
