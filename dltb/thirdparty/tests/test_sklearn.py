from unittest import TestCase, skipIf
import importlib

from ...datasource import Datasource


@skipIf(not importlib.util.find_spec('sklearn'), "sklearn not installed")
class TestSklearn(TestCase):

    def setUp(self):
        # importing sklearn should implicitly import ..sklearn
        # and register resources like datasources:
        importlib.import_module('sklearn')

    def test_lfw1(self):
        lfw = Datasource['lfw-sklearn']
        lfw._min_faces_per_person = 70  # FIXME[hack]
        lfw.prepare()
        self.assertEqual(len(lfw), 1288)

        data = lfw[1]
        self.assertEqual(data.shape, (62, 47, 3))
        self.assertEqual(data.label, 6)
        self.assertEqual(data.label['text'], 'Tony Blair')

