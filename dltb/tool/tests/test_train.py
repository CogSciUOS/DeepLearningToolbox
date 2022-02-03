"""Tests for the :py:class:`Trainer` class.
"""

# standard imports
from unittest import TestCase

# toolbox imports
from dltb.tool.train import Trainer, Trainable
from dltb.datasource import Noise


class MockupTrainee(Trainable):
    """A dummy trainee for testing the training API
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prepared = False
        self.cleaned = False
        self.restored = False
        self.stored = 0
        self.trained_single = 0
        self.trained_batch = 0

    def prepare_training(self, restore: bool = False) -> None:
        """allocate resources required for training.
        """
        self.prepared = True

    def clean_training(self) -> None:
        """free up resources required for training
        """
        self.cleaned = True

    def train_single(self, example) -> None:
        """Train this `Trainable` on a single example.
        """
        self.trained_single += 1

    def train_batch(self, batch, epoch: int) -> None:
        """Train this `Trainable` on a batch of data.
        """
        self.trained_batch += 1

    def store_checkpoint(self) -> None:
        """Store current state in a checkpoint.
        """
        self.stored += 1

    def restore_from_checkpoint(self) -> None:
        """Restore state from a checkpoint.
        Restored information should include training_step (epoch/batch)
        """
        self.restored = True


class TestTrainer(TestCase):
    """Tests for the :py:class:`Trainer` class.
    """

    def test_trainer_prepare(self) -> None:
        """Test the :py:class:`Trainer` preparation.
        """
        trainer = Trainer()
        self.assertTrue(trainer.prepared)

    def test_training(self) -> None:
        """Test that training flags are set correctly during training.
        """
        trainee = MockupTrainee()
        self.assertFalse(trainee.prepared)
        self.assertFalse(trainee.cleaned)
        self.assertFalse(trainee.restored)
        self.assertEqual(trainee.trained_single, 0)
        self.assertEqual(trainee.trained_batch, 0)

        datasource = Noise(length=1000)
        trainer = Trainer(trainee=trainee, training_data=datasource)
        trainer.train()
        # self.assertTrue(trainee.prepared)
        # self.assertTrue(trainee.cleaned)
        # self.assertTrue(trainee.restored)
        self.assertEqual(trainee.trained_single, 0)
        self.assertEqual(trainee.trained_batch, 8)  # 8 = 1000 // 128 + rest
