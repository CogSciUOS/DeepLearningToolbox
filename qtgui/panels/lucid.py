'''
File: internals.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
'''

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QSpinBox,
                             QComboBox, QListWidget, QLineEdit,
                             QVBoxLayout, QHBoxLayout, QFormLayout)
from .panel import Panel
from qtgui.utils import QImageView

import numpy as np
import tensorflow as tf

from network import loader

from tools.lucid import Engine, EngineObserver, EngineChange
from controller import LucidController

class LucidPanel(Panel, EngineObserver):
    '''A Panel displaying lucid visualizations.

    https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/tutorial.ipynb#scrollTo=8hrCwdxhcUHn

    Lucid splits visualizations into a few components which you can
    fiddle with completely indpendently:

    * objectives -- What do you want the model to visualize?
    
    * parameterization -- How do you describe the image?

    * transforms -- What transformations do you want your
      visualization to be robust to?


    Objectives
    ----------
    1. Let's visualize another neuron using a more explicit objective:
        obj = objectives.channel("mixed4a_pre_relu", 465)
        
    2. Or we could do something weirder:
       (Technically, objectives are a class that implements addition.)

         channel = lambda n: objectives.channel("mixed4a_pre_relu", n)
         obj = channel(476) + channel(465)

    Transformation Robustness
    -------------------------

    Recomended reading: The Feature Visualization article's section
    titled The Enemy of Feature Visualization discusion of
    "Transformation Robustness." In particular, there's an interactive
    diagram that allows you to easily explore how different kinds of
    transformation robustness effects visualizations.

    1. No transformation robustness
        transforms = []
    2. Jitter 2
        transforms = [ transform.jitter(2) ]
    3. Breaking out all the stops
        transforms = [
           transform.pad(16),
           transform.jitter(8),
           transform.random_scale([n/100. for n in range(80, 120)]),
           transform.random_rotate(range(-10,10) + range(-5,5) + 10*range(-2,2)),
           transform.jitter(2)
         ]

    Experimenting with parameterization
    -----------------------------------

    Recomended reading: The Feature Visualization article's section on
    Preconditioning and Parameterization

    1. Using alternate parameterizations is one of the primary
       ingredients for  effective visualization

         param_f = lambda: param.image(128, fft=False, decorrelate=False)

    2. param_f = lambda: param.image(128, fft=True, decorrelate=True)
    '''

    def __init__(self, parent=None):
        """Construct a new :py:class:`LucidPanel`.
        """
        super().__init__(parent)
        self.initUI()


    def setController(self, controller: LucidController) -> None:
        """Set the :py:class:`LucidController` for this
        :py:class:`LucidPanel`. The controller is required to
        interact with the underlying lucid engine.

        Arguments
        ---------
        controller: LucidController
            The controller to use.
        """
        engine = None if controller is None else controller.engine

        self._networks.clear()
        if engine is not None:
            self._networks.addItems(loader.lucid_names())

        self._button.clicked.connect(controller.onMaximize)
        self._buttonAll.clicked.connect(controller.onMaximizeMulti)
        self._buttonStop.clicked.connect(controller.onStop)
        self._networks.activated[str].connect(controller.onModelSelected)

        def slot(value: int): engine.unit = value
        self._unitNumber.valueChanged.connect(slot)

        self._controller = controller
        self._modelView.setEngine(engine)
        self.observe(engine, None)
        engine.notify(self)

    def initUI(self):
        
        self._imageView = QImageView()

        self._layerName = QLabel()
        self._unitNumber = QSpinBox()
        self._numberOfUnits = QLabel()
        self._unitID = QLabel()

        self._button = QPushButton("Run")
        self._buttonAll = QPushButton("Run all")
        self._buttonStop = QPushButton("Stop")

        self._networks = QComboBox()

        self._modelView = QLucidModelView()

        self.layoutUI()

    def layoutUI(self):
        layout = QVBoxLayout(self)
        layout.addWidget(self._imageView)

        h1 = QHBoxLayout()
        v1 = QVBoxLayout()
        v1.addWidget(self._networks)

        h2 = QHBoxLayout()
        h2.addWidget(self._layerName)
        h2.addWidget(self._unitNumber)
        h2.addWidget(QLabel("/"))
        h2.addWidget(self._numberOfUnits)
        h2.addWidget(self._unitID)
        v1.addLayout(h2)
        
        h1.addLayout(v1)
        h1.addWidget(self._button)
        h1.addWidget(self._buttonAll)
        h1.addWidget(self._buttonStop)
        h1.addStretch()
        
        layout.addLayout(h1)
        layout.addWidget(self._modelView)


    def engineChanged(self, engine: Engine, info: EngineChange) -> None:
        """Respond to change in the activation maximization Engine.

        Parameters
        ----------
        engine: Engine
            Engine which changed (since we could observe multiple ones)
        info: ConfigChange
            Object for communicating which aspect of the engine changed.
        """
        if info.model_changed:
            new_index = self._networks.findText(engine.model_name)
            self._networks.setCurrentIndex(new_index)
            self._modelView.setModel(engine.model_name, engine.model)

        if info.unit_changed:
            self._layerName.setText(engine.layer)
            self._numberOfUnits.setText(str(engine.layer_units))
            self._modelView.setLayer(engine.layer)

            if engine.layer_units:
                self._unitNumber.setRange(0, engine.layer_units-1)
                self._unitNumber.setValue(engine.unit)
                self._unitNumber.setEnabled(True)
            else:
                self._unitNumber.setRange(0,0)
                self._unitNumber.setValue(0)
                self._unitNumber.setEnabled(False)
            self._unitID.setText(engine.unit_id)

        if info.engine_changed:
            self._button.setEnabled(not engine.running)
            self._buttonAll.setEnabled(not engine.running)
            self._buttonStop.setEnabled(engine.running)
            self._imageView.setImage(None if engine.image is None
                                     else engine.image[0])





from lucid.modelzoo.vision_base import Model as LucidModel


class QLucidModelView(QWidget):
    """A Widget to display properties of a Lucid model.  A Lucid model
    represents a specific network trained on a specific dataset.

    
    Attributes
    ----------
    _engine: Engine
        A Lucid engine that is used as a wrapper to the actual Lucid
        classes.

    Graphical elements
    ------------------
    _dataset: QLabel
        A label displaying the name of the dataset on which the
        lucid model was trained.

    _image_shape: QLabel
        A label displaying the input shape of the network
        (without batch axis).
        
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def setEngine(self, engine: Engine) -> None:
        def slot(text):
            try: engine.layer = text[:text.index(':')]
            except ValueError: engine.layer = None
        self._layers.currentTextChanged.connect(slot)


    def initUI(self):
        """Initialize the graphical components of the user interface.
        This will create the widgets and call :py:meth:`layoutUI` to
        arrange them.
        """

        # lucid.modelzoo.vision_base.Model

        # create_input: function
        #    Create input tensor.
        
        # dataset: str
        #
        self._dataset = QLabel()

        # image_shape: List[int]
        #     [227, 227, 3]
        self._image_shape = QLabel()

        # image_value_range: tuple
        #     ()
        self._image_value_range = QLabel()

        # import_graph: function
        #    import_graph(t_input=None, scope='import', forget_xy_shape=True)

        # input_name: str
        #    'Placeholder'
        self._input_name = QLabel()

        # is_BGR: bool
        self._is_BGR = QLabel()

        # labels: None
        self._labels = QListWidget()

        # labels_path: str
        #     'gs://modelzoo/labels/ImageNet_standard.txt'
        self._labels_path = QLabel()

        # layers: List[dict]
        #     {'type': 'conv', 'name': 'concat_2', 'size': 256}
        self._layers = QListWidget()

        # load_graphdef: function

        # model_path: str
        #     'gs://modelzoo/vision/other_models/AlexNet.pb'
        self._model_path = QLabel()      

        # mro: function
        #     return a type's method resolution order (as a list)

        # post_import: function
        #      post_import(scope)

        # show_graph: function
        
        self.layoutUI()

    def layoutUI(self) -> None:
        """Layout the graphical components of this QLucidModelView.
        """
        layout = QVBoxLayout(self)
        l1 = QFormLayout()
        l1.addRow("Dataset", self._dataset)
        l1.addRow("Image shape", self._image_shape)
        l1.addRow("Image value range", self._image_value_range)
        l1.addRow("Input name", self._input_name)
        l1.addRow("Is BGR", self._is_BGR)
        layout.addLayout(l1)
        
        l1 = QHBoxLayout()
        l2 = QVBoxLayout()
        l2.addWidget(QLabel("Layers:"))
        l2.addWidget(self._layers) # "Layers"
        l2.addWidget(self._model_path) # "Model path"
        l1.addLayout(l2)
        l2 = QVBoxLayout()
        l2.addWidget(QLabel("Labels:"))
        l2.addWidget(self._labels) # "Labels"
        l2.addWidget(self._labels_path) # "Labels path",
        l1.addLayout(l2)
        layout.addLayout(l1)

    def setModel(self, name: str, model: LucidModel) -> None:
        """Set a Lucid model. This will display the properties of the
        Lucid model in this QLucidModelView.
        """
        self._layers.clear()
        if model is None:
            self._dataset.setText("")
            self._image_shape.setText("")
            self._image_value_range.setText("")
            self._input_name.setText("")
            self._is_BGR.setText("")
            #self._labels.setText("")
            self._labels_path.setText("")
            self._model_path.setText("")
        else:
            self._dataset.setText(f"{model.dataset}")
            self._image_shape.setText(f"{model.image_shape}")
            self._image_value_range.setText(f"{model.image_value_range}")
            self._input_name.setText(f"{model.input_name}")
            self._is_BGR.setText(f"{getattr(model,'is_BGR','')}")
            #self._labels.setText("{model.labels}")
            self._labels_path.setText(f"{model.labels_path}")
            self._layers.addItems([f"{l['name']}: {l['type']}[{l['size']}]"
                                   for l in model.layers])
            self._model_path.setText(f"{model.model_path}")

    def setLayer(self, name: str) -> None:
        """Select the currently selected layer.
        """
        matches = self._layers.findItems(f"{name}: ",
                                         QtCore.Qt.MatchStartsWith)
        try:
            self._layers.setCurrentItem(matches[0])
        except:
            pass
