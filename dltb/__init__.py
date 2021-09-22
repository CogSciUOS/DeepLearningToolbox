"""Central initialization file of the Deep Learning ToolBox (dltb).

This is the first file that will be automatically imported when the
Deep Learning Toolbox is used.  Importing this file should be fast,
and avoid anything may not be necessary and especially actions that
could be error prone.

Hence the imports in this file (and files imported by the file) should
be reduced to the bare minimum required to ensure that basic
functionality is guaranteed to work and that further functionality can
be added later on.

Files imported by this file should not themself import any other files
from the toolbox, as this will implicitly trigger the import of this
file and hence cause an infinite loop

The use of thirdparty modules should be avaoided if possible.  If
there are reasons to use thirdparty modules, there should be
sufficient checks and fallback solutions, if those modules are not
installed.

"""
from pathlib import Path
import os
import sys
import importlib

# The config module is the central interface for configuring the
# Deep Learning ToolBox. It is imported as one of the first modules,
# as it allows to influence the following initionalization process.
from .config import config

# util.importer2 will patch the import machinery, allowing to adapt the
# import process of certain modules.  This should be done early during
# the initialization process, before the critical modules have been
# iported.
from .util.importer2 import import_interceptor

# Importing thirdparty adapts the import of some third-party libraries
# (like keras).  This should be done before those libraries are
# imported for the first time.
from . import thirdparty

# thirdparty.datasource registers some additional Datasources.
from .thirdparty import datasource


# Adapt sys.path: Make sure that the `dltb` directory is in sys.path,
# but no subdirectory (this may happen if one of the modules is run as
# a script).
DLTB_DIRECTORY = Path(__file__).resolve().parent
ROOT_DIRECTORY = DLTB_DIRECTORY.parent

sys.path = [directory for directory in sys.path
            if not directory.startswith(str(ROOT_DIRECTORY / 'dltb'))]
if DLTB_DIRECTORY not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

# load local configuration module: the file local.py may contain
# additional configurations that overwrite the default values from
# the config module.
if (DLTB_DIRECTORY / 'local.py').is_file():
    importlib.import_module('.local', package=__name__)


if __name__ == '__main__':
    # The file is run as a script. This should usually not be possible,
    # as we are using relative imports (which will fail befor this code
    # is reached).  Nevertheless, in case this code is executed despite
    # of being impossible, we will issue some feedback to the user:
    print("To run the Deep Learning Toolbox (dltb), call 'python -m dltb' ")
    print("or use the script `dl-toolbox.py`.")
