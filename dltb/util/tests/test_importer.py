"""Testsuite for the importer API.
"""

# standard imports
from unittest import TestCase
import sys
import importlib

# toolbox imports
from dltb.util.importer import ImportInterceptor


class ImporterTest(TestCase):
    """Tests for the :py:mod:`dltb.util.importer` module.
    """

    def test_aliases(self) -> None:
        """Test the use of an import alias.
        """
        dummy_module = sys
        module_name = 'test_aliases1'
        import_interceptor = ImportInterceptor()
        import_interceptor.add_alias(module_name, dummy_module)

        self.assertFalse(module_name in sys.modules)
        module = importlib.import_module(module_name)
        self.assertTrue(module_name in sys.modules)
        self.assertTrue(module is dummy_module)

    def test_import_hooks(self) -> None:
        """Test the pre- and postimport hooks.
        """
        dummy_module = sys
        # dummy name that should not exist
        module_name = 'test_import_hooks1'
        import_interceptor = ImportInterceptor()
        pre_name = None
        post_name = None

        def pre_patch(fullname: str, path, target=None):
            # pylint: disable=unused-argument
            nonlocal pre_name
            pre_name = fullname
            # Use the 'sys' module as module_name
            sys.modules[fullname] = dummy_module

        def post_patch(module):
            nonlocal post_name
            post_name = module.__name__

        import_interceptor.add_preimport_hook(module_name, pre_patch)
        import_interceptor.add_postimport_hook(module_name, post_patch)

        self.assertFalse(module_name in sys.modules)
        module = importlib.import_module(module_name)
        self.assertEqual(module, dummy_module)
        self.assertEqual(pre_name, module_name)
        self.assertEqual(post_name, dummy_module.__name__)
        self.assertTrue(module_name in sys.modules)
        self.assertEqual(module.__name__, dummy_module.__name__)

    def test_import_dependencies(self) -> None:
        """Test import dependencies.
        """
        dummy_module = sys
        module_name = 'test_import_dependencies1'
        pre_module_name = 'test_import_dependencies1'
        post_module_name = 'test_import_dependencies1'

        import_interceptor = ImportInterceptor()
        import_interceptor.add_alias(module_name, dummy_module)
        import_interceptor.add_alias(pre_module_name, dummy_module)
        import_interceptor.add_alias(post_module_name, dummy_module)

        import_interceptor.add_preimport_depency(module_name,
                                                 pre_module_name)
        import_interceptor.add_postimport_depency(module_name,
                                                  post_module_name)

        self.assertFalse(module_name in sys.modules)
        self.assertFalse(pre_module_name in sys.modules)
        self.assertFalse(post_module_name in sys.modules)
        module = importlib.import_module(module_name)
        self.assertEqual(module, dummy_module)
        self.assertTrue(module_name in sys.modules)
        self.assertTrue(pre_module_name in sys.modules)
        self.assertTrue(post_module_name in sys.modules)
