'''
File: internals.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
'''

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QLabel, QPushButton, QLineEdit, QVBoxLayout)
from .panel import Panel
from qtgui.utils import QImageView

import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform



class LucidPanel(Panel):
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
        super().__init__(parent)
        self._modules = {}
        
        self.initUI()

        self._model = models.InceptionV1()
        self._model.load_graphdef()

    def initUI(self):
        layout = QVBoxLayout(self)
        
        self._imageView = QImageView()
        layout.addWidget(self._imageView)

        self._unitName = QLineEdit()
        self._unitName.setText("mixed4a_pre_relu:476")
        layout.addWidget(self._unitName)

        self._button = QPushButton("Run Lucid")
        self._button.clicked.connect(self.onButtonClicked2)
        layout.addWidget(self._button)

    def onButtonClicked(self):
        image = render.render_vis(self._model, self._unitName.text())
        self._imageView.setImage(image[0])

    def onButtonClicked2(self):
        self._button.setEnabled(False)
        obj = objectives.channel("mixed4a_pre_relu", 465)
        image =  render.render_vis(self._model, obj)
        self._imageView.setImage(image[0])
        self._button.setEnabled(True)
