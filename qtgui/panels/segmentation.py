# Generic imports
import logging

import numpy as np


# Qt imports
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QSplitter

# toolbox imports
from toolbox import Toolbox
from network import Network
from datasource import Datasource, Data

# GUI imports
from .panel import Panel
from ..utils import QObserver, protect
from ..widgets.data import QDataView
from ..widgets.image import QImageView
from ..widgets.network import QLayerSelector, QNetworkSelector
from ..widgets.classesview import QClassesView
from ..widgets.datasource import QDatasourceNavigator

# logging
LOG = logging.getLogger(__name__)

from experiments.deeplab import DeepLabModel
from experiments.pascal import LABEL_NAMES, FULL_COLOR_MAP, label_to_color_image

class SegmentationPanel(Panel, QObserver, qobservables={
        Toolbox: {'input_changed'}}):
    """
    """

    def __init__(self, toolbox: Toolbox = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.setToolbox(toolbox)
        self._model = DeepLabModel('mobilenetv2_coco_voctrainaug')

    def _initUI(self):
        """Initialize all UI elements. These are
        * The ``QActivationView`` showing the unit activations on the left
        * The ``QImageView`` showing the current input image
        * A ``QDatasourceNavigator`` to show datasource navigation controls
        * A ``QLayerSelector``, a widget to select a layer in a network
        * A ``QDataInfoBox`` to display information about the input
        """

        #
        # Input data
        #

        # QImageView: a widget to display the input data
        self._imageView1 = QImageView()
        self.addAttributePropagation(Toolbox, self._imageView1)
        self._imageView2 = QImageView()
        self.addAttributePropagation(Toolbox, self._imageView2)
        self._imageView3 = QImageView()
        self.addAttributePropagation(Toolbox, self._imageView3)

        # QDatasourceNavigator: navigate through the datasource
        self._datasourceNavigator = QDatasourceNavigator()
        self.addAttributePropagation(Toolbox, self._datasourceNavigator)

    def _layoutUI(self):
        inputLayout = QVBoxLayout()

        imageLayout = QHBoxLayout()
        imageLayout.addWidget(self._imageView1)
        imageLayout.addWidget(self._imageView2)
        imageLayout.addWidget(self._imageView3)

        inputLayout.addLayout(imageLayout)
        inputLayout.addWidget(self._datasourceNavigator)
        inputLayout.addStretch()

        self.setLayout(inputLayout)

    def toolbox_changed(self, toolbox: Toolbox, change: Toolbox.Change):
        # pylint: disable=invalid-name
        """React to changes of the toolbox. The :py:class:`ActivationsPanel`
        will reflect two types of changes:
        (1) `input_change`: new input data for the toolbox will be used
            as input for the underlying :py:class:`ActivationEngine`, and
        (2) `datasource_changed`: the toolbox datasource will be used
            for selecting inputs by the datasource navigator of them
            :py:class:`ActivationsPanel`.
        """
        LOG.debug("ActivationsPanel.toolbox_changed: %s", change)
        if change.input_changed:
            data = toolbox.input_data if toolbox is not None else None
            self._imageView1.setData(data)

            import PIL.Image
            if data is not None and data.data is not None:
                original_im = PIL.Image.fromarray(data.data.astype('uint8'),
                                                  'RGB')
                resized_im, seg_map = self._model.run(original_im)
                seg_image = label_to_color_image(seg_map).astype(np.uint8)
                print("resized:", type(resized_im), resized_im.size)
                print("seg_map:", type(seg_map), seg_map.shape)
                data.add_attribute('seg_image', seg_image)
                self._imageView2.setData(data, 'seg_image')
            else:
                self._imageView2.setData(data)

            self._imageView3.setData(data)
