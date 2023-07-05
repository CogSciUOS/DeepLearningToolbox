"""Testsuite for the `dltb.base.video` module.
"""

# standard imports
import unittest
from pathlib import Path

# thirdparty imports
import numpy as np

# toolbox imports
from dltb.base.image import Image
from dltb.base.video import VideoDirectory, RandomReader


class TestVideo(unittest.TestCase):
    """Tests the classes :py:class:`VideoDirectory`.
    """

    def test_video_directory(self) -> None:
        """Test :py:class:`Size` type.
        """
        video_directory = VideoDirectory(directory='test')
        name = video_directory.filename_for_frame(13)
        self.assertIsInstance(name, Path)
        self.assertEqual(str(name), 'test/frame_13.jpg')

    def test_random_reader(self) -> None:
        """Test the :py:class:`RandomReader` class.
        """
        size = (100, 50)
        channels = 3
        reader = RandomReader(size=size, channels=channels)
        frame = next(reader)

        # sanity check for Image frame
        self.assertIsInstance(frame, Image)
        self.assertTrue(hasattr(frame, 'array'))
        self.assertTrue(hasattr(frame, 'shape'))
        self.assertEqual(frame.array.shape, frame.shape)

        self.assertTrue(hasattr(frame, 'size'))
        self.assertEqual(frame.size, size)

        self.assertTrue(hasattr(frame, 'channels'))
        self.assertEqual(frame.channels, channels)

    def test_random_reader_raw(self) -> None:
        """Test the :py:class:`RandomReader` class in raw mode.
        """
        size = (100, 50)
        channels = 3
        reader = RandomReader(size=size, channels=channels, raw_mode=True)
        frame = next(reader)

        # sanity check for numpy frame
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[0], size[1])
        self.assertEqual(frame.shape[1], size[0])
        self.assertEqual(frame.shape[2], channels)
