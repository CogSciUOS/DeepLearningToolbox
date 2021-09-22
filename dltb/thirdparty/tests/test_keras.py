"""Test suite for Keras.
"""

# standard imports
from unittest import TestCase  #, skipIf
import sys
import importlib

# toolbox imports
import dltb.thirdparty as thirdparty

# FIXME[bug]: importlib.util.find_spec('keras') currently loads the
# tnesorflow.keras module as 'keras', triggered by the pre-import hook on the
# 'keras' module! (see bug description in dltb.util.importer2).
# Hence using importlib.util.find_spec('keras') to test if keras is available
# does not work.  Hence conditional testing (@skipIf) is currently not
# possible, leading to an error, when 'keras' is not installed.

# @skipIf(not importlib.util.find_spec('keras') and
#        not importlib.util.find_spec('tensorflow'), "keras not installed")
class TestKeras(TestCase):
    """Test suite for Keras.
    """

    def test_import_keras_io(self):
        # make sure no keras module has been loaded yet ...
        if 'keras' in sys.modules:
            del sys.modules['keras']
        importlib.invalidate_caches()

        thirdparty.prefer_tensorflow_keras = False
        thirdparty.import_interceptor.\
            add_preimport_hook('keras', thirdparty.patch_keras_import)

        self.assertFalse('keras' in sys.modules)
        keras = importlib.import_module('keras')
        self.assertTrue('keras' in sys.modules)
        self.assertEqual(keras.__name__, 'keras')

    def test_import_keras_tensorflow(self):
        # make sure no keras module has been loaded yet ...
        if 'keras' in sys.modules:
            del sys.modules['keras']
        importlib.invalidate_caches()

        thirdparty.prefer_tensorflow_keras = True
        thirdparty.import_interceptor.\
            add_preimport_hook('keras', thirdparty.patch_keras_import)

        self.assertFalse('keras' in sys.modules)
        keras = importlib.import_module('keras')
        self.assertTrue('keras' in sys.modules)
        self.assertEqual(keras.__name__, 'tensorflow.keras')


class TestKerasDatasource(TestCase):

    def setUp(self):
        from dltb.thirdparty.keras.datasource import KerasDatasource
        self.datasource = KerasDatasource('mnist', 'test')

    def test_len(self):
        self.datasource.prepare()
        self.assertEqual(len(self.datasource), 10000)
