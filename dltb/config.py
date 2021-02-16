"""Configuration of the Deep Learning Toolbox (dltb).
"""


# FIXME[hack]: develop some config mechanism
class Config:

    def set_default_value(self, name: str, value) -> None:
        if not hasattr(self, name):
            setattr(self, name, value)


config = Config()
config.warn_missing_dependencies = False

config.thirdparty_info = False


# FIXME[hack]: put this into some configuration file
config.github_directory = '/space/home/ulf/github'


# model_directory: str
#    A directory for storing models (architecture and weights)
#    Mainly used for pretrained models, downloaded from some
#    third-party repository.
# FIXME[hack]: put this into some configuration file
config.model_directory = '/space/home/ulf/models'
