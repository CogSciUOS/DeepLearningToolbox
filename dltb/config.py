"""Configuration of the Deep Learning Toolbox (dltb).
"""
import os


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
config.base_directory = os.path.dirname(os.path.dirname(__file__))

# FIXME[hack]: put this into some configuration file
config.github_directory = '/space/home/ulf/github'
config.github_directory = '/work/krumnack/git/'


# model_directory: str
#    A directory for storing models (architecture and weights)
#    Mainly used for pretrained models, downloaded from some
#    third-party repository.
# FIXME[hack]: put this into some configuration file
config.model_directory = '/space/home/ulf/models'
config.model_directory = '/work/krumnack/models'

config.models_directory = os.path.join(config.base_directory, 'models')

#
# Global Flags
#

# use_cpu: Should we use CPU (even if GPU is available)?
config.use_cpu = False
