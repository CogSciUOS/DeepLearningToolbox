from os import getcwd
from os.path import dirname, join
import sys

NETWORK_PACKAGE_DIRECTORY = dirname(dirname(__file__))
BASE_DIRECTORY = dirname(NETWORK_PACKAGE_DIRECTORY)
MODELS_DIRECTORY = join(BASE_DIRECTORY, 'models')

# The following lines will make the unittests work even when
# called from non-standard directories and with the network
# package not being in sys.path.
#
# Having NETWORK_PACKAGE_DIRECTORY in sys.path will hide global
# packages with same name as our networks and hence prevent
# importing those packages, like "keras".
# Hence we remove NETWORK_PACKAGE_DIRECTORY from sys.path ...
if getcwd() == NETWORK_PACKAGE_DIRECTORY and '' in sys.path: sys.path.remove('')
if NETWORK_PACKAGE_DIRECTORY in sys.path: sys.path.remove(NETWORK_PACKAGE_DIRECTORY)
# ... and make sure that the BASE_DIRECTOR is included.
BASE_DIRECTORY in sys.path or sys.path.insert(0,BASE_DIRECTORY)
