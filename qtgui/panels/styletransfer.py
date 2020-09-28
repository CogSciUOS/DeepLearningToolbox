"""
File: styltransfer.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
"""

# Generic imports
import logging
import os

# third party imports
import imageio
import tensorflow as tf

# Qt imports
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton

# toolbox imports
import dltb
from models.styletransfer import StyletransferTool

# GUI imports
from .panel import Panel
from ..utils import protect
from ..widgets.image import QImageView

# logging
LOG = logging.getLogger(__name__)


class StyletransferData:
    """Provide data for style transfer.

    Attributes
    ----------

    _content: dict
        mapping of content names to filenames

    _styles: dict
        mapping of style names to filenames
    """

    venice_url = \
        ('https://facts.uk/wp-content/uploads/2020/02/'
         'facts-about-Venice-Italy-1920x1080.jpg')

    starry_night_url = \
        ('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/'
         'Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/'
         '1920px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')

    def __init__(self) -> None:
        self._styletransfer_dir = \
            os.path.join(dltb.directories['data'], 'styletransfer')
        self._content = {}
        self._content_dir = os.path.join(self._styletransfer_dir, 'content')
        self._styles = {}
        self._style_dir = os.path.join(self._styletransfer_dir, 'style')

    def add_content_image(self, name: str, url) -> None:
        suffix = os.path.splitext(url)[1]
        filename = os.path.join(self._content_dir, name) + suffix
        self._content[name] = tf.keras.utils.get_file(filename, url)

    def add_style_image(self, name: str, url) -> None:
        suffix = os.path.splitext(url)[1]
        filename = os.path.join(self._style_dir, name) + suffix
        self._styles[name] = tf.keras.utils.get_file(filename, url)

    def prepare(self):
        os.makedirs(self._style_dir, exist_ok=True)
        self.add_style_image('starry_night', self.starry_night_url)
        print(f"starry_night: {self._styles['starry_night']}")

        os.makedirs(self._content_dir, exist_ok=True)
        self.add_content_image('venice', self.venice_url)
        print(f"venice: {self._content['venice']}")


class StyletransferPanel(Panel):

    def __init__(self, **kwargs) -> None:
        """Initialize the :py:class:`StyletransferPanel`.
        """
        super().__init__(**kwargs)
        self._engine = None
        self._initUI()
        self._initLayout()

    def _initUI(self) -> None:
        self._contentView = QImageView()
        self._styleView = QImageView()
        self._imageView = QImageView()
        self._button = QPushButton("Initialize")
        self._button.clicked.connect(self.onButtonClicked)

    def _initLayout(self) -> None:
        inputs = QHBoxLayout()
        inputs.addWidget(self._contentView)
        inputs.addWidget(self._styleView)
        rows = QVBoxLayout()
        rows.addLayout(inputs)
        rows.addWidget(self._imageView)
        rows.addWidget(self._button)
        self.setLayout(rows)

    def setStyletransfer(self, engine) -> None:
        if self._engine is not None:
            self._imageView.unobserve(self._engine)
        self._engine = engine
        if self._engine is not None:
            self._imageView.observe(self._engine)
            self._contentView.setImage(self._engine.content)
            self._styleView.setImage(self._engine.style)
            self._button.setText("Start/Stop")
        else:
            self._button.setText("Initialize")

    @protect
    def onButtonClicked(self, checked: bool) -> None:
        if self._engine is None:
            self._initEngine()
        else:
            if self._engine.looping:
                self._engine.stop()
            else:
                self._engine.loop(threaded=True)

    def _initEngine(self) -> None:
        # assert tf.__version__ >= '2.0.0'
        tf.compat.v1.enable_eager_execution()

        print(f"TensorFlow: is_gpu_available: {tf.test.is_gpu_available()}")
        # cuda_only=False, min_cuda_compute_capability=None

        data = StyletransferData()
        data.prepare()

        content_layers = ['block5_conv2']

        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

        tool = StyletransferTool(style_layers=style_layers,
                                 content_layers=content_layers)
        tool.content = imageio.imread(data._content['venice'])
        tool.style = imageio.imread(data._styles['starry_night'])
        tool.reset()

        self.setStyletransfer(tool)
