"""Test suite for Keras.
"""

# standard imports
from unittest import TestCase, skip, skipUnless
import sys
import importlib

# toolbox imports
from dltb.config import config
from dltb.util.importer import importable, import_interceptor
from dltb.thirdparty import keras_register


@skipUnless(importable('keras') or importable('tensorflow'),
            "Keras is not installed")
class TestKeras(TestCase):
    """Test suite for Keras.
    """

    @staticmethod
    def _unimport_keras() -> None:
        if 'keras' in sys.modules:
            if 'tensorflow' in sys.modules and \
                    sys.modules['tensorflow.keras'] is sys.modules['keras']:
                sys.modules.pop('keras')
            else:
                del sys.modules['keras']
        importlib.invalidate_caches()
    
    @staticmethod
    def _ensure_import_hook() -> None:
        # The preimport_hook should already be installed by importing
        # dltb.thirdparty.keras_register. However, it will be
        # automatically removed after the import (assuming that there
        # will be no second import, which is usually sensible, but
        # wrong for this testsuite ...)
        if 'keras' not in import_interceptor._preimport_hooks:
            import_interceptor.\
                add_preimport_hook('keras', keras_register.patch_keras_import)
    

    @skipUnless(importable('keras'), "Keras.io is not available")
    def test_import_keras_io(self):
        """Import the old stand alone Keras (Keras.io).

        Note: with newer Numpy (>=1.20), this may issue some deprecation
        warnings.
        """
        # make sure no keras module has been loaded yet ...
        self._unimport_keras()
        self.assertNotIn('keras', sys.modules)

        config.keras_prefer_tensorflow = False
        self._ensure_import_hook()
        self.assertIn('keras', import_interceptor._preimport_hooks)

        self.assertFalse('keras' in sys.modules)
        keras = importlib.import_module('keras')
        self.assertTrue('keras' in sys.modules)
        self.assertEqual(keras.__name__, 'keras')

    @skipUnless(importable('tensorflow'), "TensorFlow is not available")
    @skip("The import interceptor is buggy - needs some fix!")
    def test_import_keras_tensorflow(self):
        # make sure no keras module has been loaded yet ...
        self._unimport_keras()
        self.assertNotIn('keras', sys.modules)

        config.keras_prefer_tensorflow = True
        self._ensure_import_hook()
        self.assertIn('keras', import_interceptor._preimport_hooks)

        # import keras (here we expect that tensorflow.keras is imported
        # as keras - however, inspecting its name should reveal its true
        # identity)
        keras = importlib.import_module('keras')

        # FIXME[bug]: there is some strange thing happening during import:
        # the module  sys.modules['tensorflow.keras'] and
        # sys.modules['tensorflow'].keras, which before the import are
        # the same, are different afterwards. Hence some asserts fail.
        self.assertIs(sys.modules['tensorflow.keras'],
                      sys.modules['tensorflow'].keras)

        if 'tensorflow' in sys.modules:
            self.assertIs(sys.modules['tensorflow.keras'],
                          sys.modules['tensorflow'].keras)

        self.assertIn('keras', sys.modules)
        self.assertEqual(keras.__name__, 'tensorflow.keras')
        # FIXME[bug]: The following fails!
        self.assertIs(sys.modules['keras'], keras)


@skipUnless(importable('tensorflow'), "Tensorflow is not installed")
class TestKerasDatasource(TestCase):

    def setUp(self):
        from dltb.thirdparty.keras.datasource import KerasDatasource
        self.datasource = KerasDatasource('mnist', 'test')

    def test_len(self):
        self.datasource.prepare()
        self.assertEqual(len(self.datasource), 10000)
