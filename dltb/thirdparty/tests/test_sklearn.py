"""Unit tests for deep learning toolbox API to the scikit-learn module.
"""

# standard imports
from unittest import TestCase, skipIf, skipUnless
import importlib

# toolbox imports
from ...datasource import Datasource


@skipIf(importlib.util.find_spec('sklearn') is None, "sklearn not installed")
class TestSklearn(TestCase):
    """Unit tests for deep learning toolbox API to the scikit-learn module.
    """

    def setUp(self):
        # importing sklearn should implicitly import ..sklearn
        # and register resources like datasources:
        importlib.import_module('sklearn')

    @skipUnless('lfw-sklearn' in Datasource,
                "Datasource 'lfw-sklearn' is not registered.")
    def test_lfw1(self):
        """Test the scikit-learn function to access the Labeled Faces in the
        Wild (LFW) dataset.

        """
        lfw = Datasource['lfw-sklearn']
        lfw._min_faces_per_person = 70  # FIXME[hack]
        lfw.prepare()
        self.assertEqual(len(lfw), 1288)

        data = lfw[1]
        self.assertEqual(data.shape, (62, 47, 3))
        self.assertEqual(data.label, 6)
        self.assertEqual(data.label['text'], 'Tony Blair')
