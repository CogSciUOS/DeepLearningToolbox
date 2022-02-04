"""Unit tests for deep learning toolbox API to the scikit-learn module.
"""

# standard imports
from unittest import TestCase, skipIf, skipUnless
import importlib

# toolbox imports
from dltb.datasource import Datasource


@skipIf(importlib.util.find_spec('sklearn') is None, "sklearn not installed")
class TestSklearn(TestCase):
    """Unit tests for deep learning toolbox API to the scikit-learn module.
    """
    # 
    importlib.import_module('dltb.thirdparty.sklearn')

    def setUp(self):
        # importing sklearn should implicitly import ..sklearn
        # and register resources like datasources:
        importlib.import_module('sklearn')

    def test_lfw_datasource(self):
        """Test if the scikit-learn "Labeled Faces in the Wild" (LFW)
        module is registered as `Datasource`.
        """
        self.assertTrue('lfw-sklearn' in Datasource)

    @skipUnless('lfw-sklearn' in Datasource,
                "Datasource 'lfw-sklearn' is not registered.")
    def test_lfw_datasource_size(self):
        """Test the scikit-learn function to access the Labeled Faces in the
        Wild (LFW) dataset.

        Remark: this test will download the LFW dataset into the
        sklearn directory if missing.  This may take some time.
        """
        lfw = Datasource['lfw-sklearn']
        self.assertTrue(lfw.prepared)
        self.assertEqual(len(lfw), 13233)

    @skipUnless('lfw-sklearn' in Datasource,
                "Datasource 'lfw-sklearn' is not registered.")
    def test_lfw_datasource_access(self):
        """Test the scikit-learn function to access the Labeled Faces in the
        Wild (LFW) dataset.
        """
        lfw = Datasource['lfw-sklearn']

        data = lfw[1]
        self.assertEqual(data.shape, (62, 47, 3))
        # self.assertEqual(data.label, 6)
        # self.assertEqual(data.label['text'], 'Tony Blair')

    @skipUnless('lfw-sklearn' in Datasource,
                "Datasource 'lfw-sklearn' is not registered.")
    def test_lfw_datasource_restrict(self):
        """Restrict the LFW datasource to classes (persons) with a
        minimal number of images.
        """
        lfw = Datasource['lfw-sklearn']

        lfw._min_faces_per_person = 70  # FIXME[hack]
        # self.assertEqual(len(lfw), 1288)
