
# standard imports
from unittest import TestCase

# toolbox imports
from dltb.base.data import Data, DataDict


class TestDataClass(TestCase):
    """Tests for the :py:class:`Data` class.
    """

    def test_instantiate_data(self):
        # Instantiating the (abstract) Data class should yield a
        # DataDict object
        data = Data()
        self.assertTrue(isinstance(data, Data))
        self.assertTrue(isinstance(data, DataDict))
