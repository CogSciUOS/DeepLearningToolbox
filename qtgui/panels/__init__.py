from util import addons

from .panel import Panel
from .activations import ActivationsPanel
from .experiments import ExperimentsPanel
from .occlusion import OcclusionPanel
from .maximization import MaximizationPanel

if addons.use('lucid'):
    from .lucid import LucidPanel

from .autoencoder import AutoencoderPanel

from .internals import InternalsPanel
from .logging import LoggingPanel

del addons
