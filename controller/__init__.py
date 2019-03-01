'''
.. moduleauthor Rasmus Diederichsen, Ulf Krumnack

.. module controller

This module includes all classes which relate to the controller
portion of the MVC pattern.

'''
from util import addons

from .base import BaseController
from .asyncrunner import AsyncRunner
from .datasource import DataSourceController
DataSourceObserver = DataSourceController.Observer  # FIXME[hack]
DataSourceChange = DataSourceController.Change  # FIXME[hack]
from .activations import ActivationsController
from .maximization import MaximizationController

if addons.use('lucid'):
    from .lucid import LucidController

del addons
