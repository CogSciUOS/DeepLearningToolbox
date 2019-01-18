"""Configuration data for the "activation maximization" (am) module.
"""
from base import Config as BaseConfig
from observer import Observer, BaseChange, change


class Config(BaseConfig):
    """Configuration data for the "activation maximization" (am) module.
    """

    class Change(BaseChange):
        ATTRIBUTES = ['network_changed', 'layer_changed', 'unit_changed',
                      'config_changed']

    _config = {
        'UNIT_INDEX': {
            'default': 0,
            'doc': 'The unit to maximize.',
            'change': 'unit_changed'
        },
        'NETWORK_KEY': {
            'default': '',
            'doc': 'The name of the network in which the unit to maximize '
                   'is found. Needs to be a key that allows to get '
                   'the network.',
            'change': 'network_changed'
        },
        'LAYER_KEY': {
            'default': '',
            'doc': 'The name of the layer in which the unit to maximize '
                   'is found. Needs to be a key that allows to get '
                   'the layer from a network object.',
            'change': 'layer_changed'
        },
        'RANDOMIZE_INPUT': {
            'default': True,
            'doc': 'maximization is initialized with random input'
                   'or flat colored image'
        },
        'ETA': {
            'default': 500,
            'doc': 'learning rate'
        },
        'BLUR_ACTIVATED': {
            'default': True,
            'doc': 'apply Gaussian blur'
        },
        'BLUR_KERNEL': {
            'default': (3, 3),
            'doc': 'Gaussian blur parameter'
        },
        'BLUR_SIGMA': {
            'default': 0,
            'doc': 'Gaussian blur parameter'
        },
        'BLUR_FREQUENCY': {
            'default': 5,
            'doc': 'how many steps between two blurs. paper used 4.'
        },

        'L2_ACTIVATED': {
            'default': True,
            'doc': 'Apply L2 decay'
        },
        'L2_LAMBDA': {
            'default': 0.000001,
            'doc': 'L2 decay parameter, totally arbitrarily chosen'
        },

        'NORM_CLIPPING_ACTIVATED': {
            'default': False,
            'doc': 'apply low norm pixel pixel clipping'
        },
        'NORM_CLIPPING_FREQUENCY': {
            'default': 25,
            'doc': 'how many steps between pixel clippings'
        },
        'NORM_PERCENTILE': {
            'default': 30,
            'doc': 'how many of the pixels are clipped'
        },

        'CONTRIBUTION_CLIPPING_ACTIVATED': {
            'default': True,
            'doc': 'Apply low contribution pixel clipping'
        },
        'CONTRIBUTION_CLIPPING_FREQUENCY': {
            'default': 50,
            'doc': 'how many steps between pixel clippings'
        },
        'CONTRIBUTION_PERCENTILE': {
            'default': 15,
            'doc': 'how many of the pixels are clipped'
        },

        'BORDER_REG_ACTIVATED': {
            'default': True,
            'doc': 'Activate border regularizer: this punishes high pixels'
                   ' the more distant they are from the image center.'
        },
        'BORDER_FACTOR': {
            'default': 0.000003,
            'doc': 'Instead of summing up the product of the actual pixel '
                   'to center distance and its value (too strong), '
                   'the effect is changed by multiplying each of the '
                   'resultant values with this factor.'
        },
        'BORDER_EXP': {
            'default': 1.5,
            'doc': 'The higher this factor, the stronger the distance '
                   'from center is punished.'
        },

        'LARGER_IMAGE': {
            'default': False,
            'doc': 'now called upscaling in thesis'
        },
        'IMAGE_SIZE_X': {
            'default': 350,
            'doc': ''
        },
        'IMAGE_SIZE_Y': {
            'default': 350,
            'doc': ''
        },

        'JITTER': {
            'default': False,
            'doc': 'Jittering causes the input image to be (randomly) '
                   'shifted by a small offset in horizontal and vertical '
                   'direction in every iteration. The idea is to achieve '
                   'a result that is invariant to such translations.'
        },
        'JITTER_STRENGTH': {
            'default': 5,
            'doc': 'The amount of jitter to use. In every iteration the '
                   'amount horizontal and vertical shift is randomly '
                   'choosen from the interval '
                   '[-_JITTER_STRENGTH, _JITTER_STRENGTH].'
        },

        'WRAP_AROUND': {
            'default': True,
            'doc': 'Pixels beyond the image borders in '
                   'upscaling/jitter are updated via wrap-around'
        },

        'LOSS_GOAL': {
            'default': 0.01,
            'doc': 'convergence parameter. relative(!) difference '
                   'between loss and last 50 losses to converge to'
        },
        'LOSS_COUNT': {
            'default': 100,
            'doc': ''
        },
        'MAX_STEPS': {
            'default': 2000,
            'doc': 'how many steps to maximally take when optimizing an image'
        },
        'TENSORBOARD_ACTIVATED': {
            'default': False,
            'doc': 'switch on and off the logging of the image and loss'
        },

        'NORMALIZE_OUTPUT': {
            'default': True,
            'doc': 'whether to save output image normalized'
        }
    }

    def __init__(self):
        super().__init__(Config.Change, 'configChanged', 'config_changed')

    # FIXME[hack]: set preferred parameters for specific networks
    @change
    def _helper_setattr(self, name, value):
        super()._helper_setattr(name, value)
        if name == 'NETWORK_KEY':
            if self.NETWORK_KEY == 'AlexNet':
                # self.LAYER_KEY = 'fc8' # default for AlexNet
                # -> does not exist ... use 'dense_3'
                # "dense_3" -> "strided_slice_3:0"
                # while 'xw_plus_b:0' -> "strided_slice_4:0"
                self.LAYER_KEY = "dense_3"
                self.UNIT_INDEX = 947


class ConfigObserver(Observer):

    def configChanged(self, config: Config, info: Config.Change) -> None:
        """Respond to change in the config.

        Parameters
        ----------
        config: Config
            Config which changed (since we could observer multiple ones)
        info: Config.Change
            Object for communicating which parts of the model changed.
        """
        print("FIXME[debug]: Bad tools.am.ConfigObserver: "
              f"configChanged({info}) is not implement")
        # pass
