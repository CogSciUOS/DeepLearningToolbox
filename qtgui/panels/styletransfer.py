"""
File: styltransfer.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
"""

# Generic imports
import logging

# third party imports
import imageio
import tensorflow as tf

# Qt imports
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton

# toolbox imports
from models.styletransfer import StyletransferData, StyletransferTool

# GUI imports
from .panel import Panel
from ..utils import protect
from ..widgets import QImageView

# logging
LOG = logging.getLogger(__name__)


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
        self._button = QPushButton("Start/Stop")
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
