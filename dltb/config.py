"""Configuration of the Deep Learning Toolbox (dltb).
"""
import os
from pathlib import Path

# FIXME[hack]: develop some config mechanism
class Config:

    def set_default_value(self, name: str, value) -> None:
        if not hasattr(self, name):
            setattr(self, name, value)


config = Config()
config.warn_missing_dependencies = False

config.thirdparty_info = False


#
# Directories
#
config.base_directory = Path(os.path.dirname(os.path.dirname(__file__)))

# FIXME[hack]: put this into some configuration file
config.github_directory = Path('/space/home/ulf/github')
config.github_directory = Path('/work/krumnack/git/')


# model_directory: str
#    A directory for storing models (architecture and weights)
#    Mainly used for pretrained models, downloaded from some
#    third-party repository.
# FIXME[hack]: put this into some configuration file
config.model_directory = Path('/space/home/ulf/models')
config.model_directory = Path('/work/krumnack/models')

config.models_directory = config.base_directory / 'models'

#config.activations_directory = Path('/space/home/ulf/activations')
config.activations_directory = \
    Path(os.environ['HOME'], 'scratch', 'activations')



#
# Global Flags
#

# use_cpu: Should we use CPU (even if GPU is available)?
config.use_cpu = False
