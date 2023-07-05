"""Testsuite for the `dltb.thirdparty.torch.network` module.
"""
# FIXME[bug]: in case of a NVML (CUDA) Driver/library version mismatch,
# running this test simply hangs (can be interrupted by kil, but not by
# ctrl+c).  Interestingly, when running the code manually (in ipython),
# the code (torch) raises a RuntimeError
# (at the line cls._network = DemoResnetNetwork()))
# Desired behavior: the test should raise an Error or be skipped.

# standard imports
import unittest

# toolbox imports
from dltb.thirdparty.torch.hub import DemoResnetNetwork
from dltb.thirdparty.torch.adversarial import IterativeGradientSignAttacker


class TestNetwork(unittest.TestCase):
    """Tests the torch `Network` implementation.
    """

    @classmethod
    def setup_class(cls) -> None:
        """Provide resources for the tests.
        """
        cls._network = DemoResnetNetwork()  # ResNet18

    @classmethod
    def teardown_class(cls) -> None:
        """Clean up resources.
        """
        del cls._network

    def test_network_settings(self):
        """Test network predictions.
        """
        # check that the model was successfully initialized
        self.assertTrue(self._network.prepared)
        self.assertIsNotNone(self._network.torch_model)

        # check the ImageNet class scheme
        self.assertEqual(self._network.lookup, 'torch')
        self.assertEqual(self._network.get_label(258), 178)
        self.assertEqual(self._network.get_label(270), 101)

    def test_network_classifier(self):
        """Test the classification result by classifying an image file.
        """

        # check the classification result
        filename = 'examples/dog2.jpg'
        torch_label = 270  # 101: Synset 'n02114548': "white wolf"
        # torch_label = 258  # 178 :Synset 'n02111889': "Samoyed"

        # compute class scores
        scores = self._network.class_scores(filename)
        self.assertEqual(scores.argmax(), torch_label)

        # test of the classifier interface (should yield the same result)
        class_identifier = self._network.classify(filename)
        self.assertEqual(class_identifier['torch'], torch_label)

    def test_adversarial_example(self):
        """Test creating an adversarial example for the network.
        """
        attacker = IterativeGradientSignAttacker()

        filename = 'examples/cat.jpg'
        target = 123

        adversarial_example = \
            attacker.attack(self._network, filename, target=target)

        class_identifier = self._network.classify(adversarial_example[0])
        self.assertEqual(class_identifier['torch'], 123)
