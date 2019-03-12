'''
.. moduleauthor Rasmus Diederichsen, Ulf Krumnack

.. module controller

This module includes all classes which relate to the controller
portion of the MVC pattern.

'''
from util import addons

#from .base import BaseController  # FIXME[old]: remove if not needed
from base import Controller as BaseController
from .asyncrunner import Runner, AsyncRunner
from .activations import ActivationsController
from .maximization import MaximizationController

if addons.use('lucid'):
    from .lucid import LucidController

del addons
