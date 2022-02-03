"""Tests for the MNIST dataset.
"""

# standard imports
from unittest import TestCase, skipUnless, SkipTest
from pathlib import Path
import shutil

# toolbox imports
from dltb import config
from dltb.datasource import Datasource
from dltb.thirdparty.datasource.mnist import MNIST


class TestMNIST(TestCase):
    """Test suite for the MNIST `Datasource`.
    """

    @classmethod
    def setUpClass(cls):  # pytest: setup_class(self)
        """Globally initialized the datasets for testing
        """
        # pylint: disable=attribute-defined-outside-init
        cls.mnist_train = MNIST(section='train', validation_split=False)
        cls.mnist_test = MNIST(section='test')

    @classmethod
    def tearDownClass(cls):  # pytest: teardown_class(self)
        """Release datasource resources.
        """
        del cls.mnist_train
        del cls.mnist_test

    def test_prepare(self):
        """Test the prepare mechanism.
        """
        self.assertTrue(self.mnist_train.prepared)
        self.assertTrue(self.mnist_test.prepared)

    def test_len(self):
        """Test the length of the datasources.
        """
        self.assertEqual(len(self.mnist_train), 60000)
        self.assertEqual(len(self.mnist_test), 10000)

    def test_data(self):
        """Test the data shape.
        """
        data = self.mnist_train[0]
        self.assertEqual(data.shape, (28, 28, 1))

    def test_implementable(self):
        """Test the `Implementable` interface.
        """
        mnist = Datasource(module='mnist', section='test')
        self.assertIsInstance(mnist, Datasource)
        self.assertIsInstance(mnist, MNIST)

    @skipUnless(config.run_slow_tests,
                reason="slow tests are disabled")
    def test_download(self):
        """Test the download mechanism.
        """
        # new directory to place the MNIST files
        mnist_directory = Path('./MNIST')

        try:
            mnist_directory.mkdir()  # may raise FileExistsError
            mnist = Datasource(module='mnist', section='test',
                               directory=mnist_directory)
            test_images = mnist_directory / 't10k-images-idx3-ubyte.gz'
            test_labels = mnist_directory / 't10k-labels-idx1-ubyte.gz'
            self.assertTrue(test_images.is_file())
            self.assertTrue(test_labels.is_file())
            self.assertEqual(len(mnist), len(self.mnist_test))
            shutil.rmtree(mnist_directory)
        except FileExistsError:
            raise SkipTest(f"Test Directory {mnist_directory} exists.") \
                from None
