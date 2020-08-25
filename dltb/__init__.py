
# Importing thirdparty will patch the machinery in order to
# adapt the import of some third-party libraries (like keras).
# This should be done before those libraries are imported for the
# first time.
from . import thirdparty
