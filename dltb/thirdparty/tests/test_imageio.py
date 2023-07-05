"""Testsuite for the `imageio` package.
"""
# standard imports
from unittest import TestCase, SkipTest, skipUnless

# thirdparty imports
try:
    import numpy  # pylint: disable=unused-import
    import imageio  # pylint: disable=unused-import
except ImportError as error:
    raise SkipTest("Third party requirements are missing") from error

# toolbox imports
from dltb.base import image, video
from dltb.util.importer import importable
from dltb.thirdparty.imageio import ImageIO, VideoReader, VideoWriter, Webcam


class ImageioTest(TestCase):
    """Tests for the :py:mod:`dltb.thirdparty.imageio` module.
    """

    def test_imagereader_registration(self) -> None:
        """Test that :py:class:`ImageIO` is registered as
        :py:class:`image.ImageReader` implementation.

        """
        reader_cls = image.ImageReader.get_implementation(module='imageio')
        self.assertEqual(reader_cls, ImageIO)

        reader = image.ImageReader(module='imageio')
        self.assertIsInstance(reader, ImageIO)

    def test_imagewriter_registration(self) -> None:
        """Test that :py:class:`ImageIO` is registered as
        :py:class:`image.ImageWriter` implementation.

        """
        writer_cls = image.ImageWriter.get_implementation(module='imageio')
        self.assertEqual(writer_cls, ImageIO)

        writer = image.ImageWriter(module='imageio')
        self.assertIsInstance(writer, ImageIO)

    @skipUnless(importable('imageio_ffmpeg'),
                "Module 'imageio_ffmpeg' is not installed.")
    def test_videoreader_registration(self) -> None:
        """Test that :py:class:`VideoReader` is registered as
        :py:class:`video.VideoReader` implementation.

        """
        reader_cls = video.VideoReader.get_implementation(module='imageio')
        self.assertEqual(reader_cls, VideoReader)

        # FIXME[bug]: passing prepare=False raises TypeError:
        #    __new__() got an unexpected keyword argument 'prepare'
        # reader = video.VideoReader(filename='test.mp4', prepare=False,
        #                            module='imageio')
        # self.assertIsInstance(reader, VideoReader)

    @skipUnless(importable('imageio_ffmpeg'),
                "Module 'imageio_ffmpeg' is not installed.")
    def test_videowriter_registration(self) -> None:
        """Test that :py:class:`VideoWriter` is registered as
        :py:class:`video.VideoWriter` implementation.

        """
        writer_cls = video.VideoWriter.get_implementation(module='imageio')
        self.assertEqual(writer_cls, VideoWriter)

        writer = video.VideoWriter(fps=25.0, size=(100, 100), frame_format='?',
                                   filename='test.mp4',
                                   module='imageio', prepare=False)
        self.assertIsInstance(writer, VideoWriter)

    @skipUnless(importable('imageio_ffmpeg'),
                "Module 'imageio_ffmpeg' is not installed.")
    def test_webcam_registration(self) -> None:
        """Test that :py:class:`Webcam` is registered as
        :py:class:`video.Webcam` implementation.

        """
        webcam_cls = video.Webcam.get_implementation(module='imageio')
        self.assertEqual(webcam_cls, Webcam)

        # FIXME[bug]: passing prepare=False raises TypeError:
        #    __new__() got an unexpected keyword argument 'prepare'
        # webcam = video.Webcam(prepare=False, module='imageio')
        # self.assertIsInstance(webcam, Webcam)
