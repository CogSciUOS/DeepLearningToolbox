"""Central initialization file of the Deep Learning ToolBox (dltb).
"""
import os
import importlib

# The config module is the central interface for configuring the
# Deep Learning ToolBox. It is imported as one of the first modules,
# as it allows to influence the following initionalization process.
from .config import config

# Importing thirdparty will patch the machinery in order to
# adapt the import of some third-party libraries (like keras).
# This should be done before those libraries are imported for the
# first time.
from . import thirdparty

# thirdparty.datasource registers some additional Datasources.
from .thirdparty import datasource


# FIXME[hack]: used by models/styltransfer
directories = {
    'data': os.path.join(os.environ.get('WORK', '.'), 'dltb-data')
}


local_config_file = os.path.join(os.path.dirname(__file__), 'local.py')
if os.path.isfile(local_config_file):
    importlib.import_module('.local', package=__name__)
