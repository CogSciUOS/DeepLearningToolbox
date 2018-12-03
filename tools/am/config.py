import copy
import numpy as np

from observer import Observer, Observable, BaseChange, change

from network import Network

class ConfigChange(BaseChange):
    ATTRIBUTES = ['network_changed', 'layer_changed', 'unit_changed',
                  'config_changed']


class ConfigObserver(Observer):

    def configChanged(self, config: 'Config', info: ConfigChange) -> None:
        """Respond to change in the config.

        Parameters
        ----------
        config : Config
            Config which changed (since we could observer multiple ones)
        info : ConfigChange
            Object for communicating which parts of the model changed.
        """
        print("HELLO base: configChanged!")
        #pass


class Config(Observable):
    """Configuration data for the "activation maximization" (am) module.

    Attributes
    ----------
    _network: Network
        FIXME[concept]: we want to remember which network was used
        for maximization, but we probably do not want access any
        properties of this network here - so probably an id/key
        would be sufficient here.

    _layer: Layer
        FIXME[concept]: some as above. Actually accessing a layer
        should be done by the engine!
        
    _LAYER_KEY: str
        The name of the layer in which the unit to maximize is found.
        Needs to be a key that allows to get the layer from a network
        object.

    _UNIT_INDEX: int

    
    _RANDOMIZE_INPUT: bool
        whether maximization is initialized with random input or flat
        colored image

    _ETA: float
        learning rate

    _BLUR_ACTIVATED: bool
        gaussian blur parameters

    _BLUR_KERNEL: tuple
    _BLUR_SIGMA: int
    _BLUR_FREQUENCY: int
        how many steps between two blurs. paper used 4.

    _L2_ACTIVATED: bool
        l2 decay parameters
        
    _L2_LAMBDA: float
        totally arbitrarily chosen
    
    _NORM_CLIPPING_ACTIVATED: bool
        low norm pixel clipping parameters

    _NORM_CLIPPING_FREQUENCY: int
        how many steps between pixel clippings

    _NORM_PERCENTILE: float
        how many of the pixels are clipped

    _CONTRIBUTION_CLIPPING_ACTIVATED: bool
        low contribution pixel clipping parameters see norm clipping
        for explanation

    _CONTRIBUTION_CLIPPING_FREQUENCY: int
    _CONTRIBUTION_PERCENTILE: float

    _BORDER_REG_ACTIVATED: bool
        Activate border regularizer: this punishes high pixels
        the more distant they are from the image center.

    _BORDER_FACTOR: float
        Instead of summing up the product of the actual pixel to
        center distance and its value (too strong), the effect is
        changed by multiplying each of the resultant values with this
        factor.

    _BORDER_EXP: float
        The higher this factor, the stronger the distance from center
        is punished.
        
    _LARGER_IMAGE: bool
        now called upscaling in thesis
    _IMAGE_SIZE_X: int
    _IMAGE_SIZE_Y: int

    _JITTER: bool
        Jittering causes the input image to be (randomly) shifted by a
        small offset in horizontal and vertical direction in every
        iteration. The idea is to achieve a result that is invariant
        to such translations.
        
    _JITTER_STRENGTH: int
        The amount of jitter to use. In every iteration the amount
        horizontal and vertical shift is randomly choosen from the
        interval [-_JITTER_STRENGTH, _JITTER_STRENGTH].

    _WRAP_AROUND: bool
        A flag indicating whether pixels beyond the image borders in
        upscaling/jitter are updated via wrap-around

    _LOSS_GOAL: float
        convergence parameters
        relative(!) difference between loss and last 50 losses to converge to

    _LOSS_COUNT: int

    _MAX_STEPS: int
        how many steps to maximally take when optimizing an image
    
    _TENSORBOARD_ACTIVATED: bool
        to easily switch on and off the logging of the image and loss

    _NORMALIZE_OUTPUT: bool
        whether to save output image normalized

    # FIXME[old]:
    RANDOM_UNIT = True
        whether a unit from the chosen layer to perform activation
    SAVE_PATH = "" # defaults to cwd
    """
    def __init__(self):
        super().__init__(ConfigChange, 'configChanged')

        self._network = None
        self._layer = None
        self._NETWORK_KEY = 'AlexNet'
        self._LAYER_KEY = 'fc8' # default for AlexNet
        self._UNIT_INDEX = 947

        self._RANDOMIZE_INPUT = True

        self._ETA = 500

        self._BLUR_ACTIVATED = True
        self._BLUR_KERNEL = (3, 3)
        self._BLUR_SIGMA = 0
        self._BLUR_FREQUENCY = 5

        self._L2_ACTIVATED = True
        self._L2_LAMBDA = 0.000001

        self._NORM_CLIPPING_ACTIVATED = False
        self._NORM_CLIPPING_FREQUENCY = 25
        self._NORM_PERCENTILE = 30

        self._CONTRIBUTION_CLIPPING_ACTIVATED = True
        self._CONTRIBUTION_CLIPPING_FREQUENCY = 50
        self._CONTRIBUTION_PERCENTILE = 15

        self._BORDER_REG_ACTIVATED = True
        self._BORDER_FACTOR = 0.000003
        self._BORDER_EXP = 1.5

        self._LARGER_IMAGE = False
        self._IMAGE_SIZE_X = 350
        self._IMAGE_SIZE_Y = 350

        self._JITTER = False
        self._JITTER_STRENGTH = 5

        self._WRAP_AROUND = True

        self._LOSS_GOAL = 0.01
        self._LOSS_COUNT= 100
        self._MAX_STEPS = 2000
        self._TENSORBOARD_ACTIVATED = False

        
        self._NORMALIZE_OUTPUT = True

    @property
    def network(self) -> Network:
        return self._network

    @change
    def _set_network_key(self, network_key: str) -> None:
        if self._NETWORK_KEY != network_key:
            self._NETWORK_KEY = network_key
            self.change(network_changed=True)

    @property
    def NETWORK_KEY(self) -> str:
        """The name of the network in which the unit to maximize is found.
        Needs to be a key that allows to get the network from a network
        object.
        """
        return self._NETWORK_KEY

    @NETWORK_KEY.setter
    def NETWORK_KEY(self, network_key: str) -> None:
        """Set the name of the network in which the unit to maximize is found.
        Needs to be a key that allows to get the network from a network
        object.
        """
        self._set_network_key(network_key)

    @change
    def _set_layer_key(self, layer_key: str) -> None:
        if self._LAYER_KEY != layer_key:
            self._LAYER_KEY = layer_key
            self.change(layer_changed=True)

    @property
    def LAYER_KEY(self) -> str:
        """The name of the layer in which the unit to maximize is found.
        Needs to be a key that allows to get the layer from a network
        object.
        """
        return self._LAYER_KEY

    @LAYER_KEY.setter
    def LAYER_KEY(self, layer_key: str) -> None:
        """Set the name of the layer in which the unit to maximize is found.
        Needs to be a key that allows to get the layer from a network
        object.
        """
        self._set_layer_key(layer_key)

        # FIXME[hack]:
        # "dense_3" -> "strided_slice_3:0"
        # while 'xw_plus_b:0' -> "strided_slice_4:0"
        #
        # 
        # The function graph.get_tensor_by_name() takes a tensor name that
        # is derived from an operatorname. There is an operator called
        # "xw_plus_b". The tensor name is than "xw_plus_b:0"
        # (where 0 refers to endpoint which is somewhat redundant)
        import tensorflow as tf
        t = tf.get_default_graph().get_tensor_by_name('xw_plus_b:0')
        print(f"*** A: {type(t)}, {t} ***")
        self._layer = t[0]
        print(f"*** B: {type(self._layer)}, {self._layer} ***")

    @change
    def _set_unit_index(self, unit_index: int) -> None:
        if self._UNIT_INDEX != unit_index:
            self._UNIT_INDEX = unit_index
            self.change(unit_changed=True)

    @property
    def UNIT_INDEX(self) -> int:
        return self._UNIT_INDEX

    @UNIT_INDEX.setter
    def UNIT_INDEX(self, unit_index: int) -> None:
        self._set_unit_index(unit_index)

    def random_unit(self):
        if self._layer is None:
            self.UNIT_INDEX = None
        else:
            self.UNIT_INDEX = np.random.randint(0, self._layer.shape[0])

    @change
    def _set_randomize_input(self, randomize_input: bool) -> None:
        if self._RANDOMIZE_INPUT != randomize_input:
            self._RANDOMIZE_INPUT = randomize_input
            self.change(config_changed=True)

    @property
    def RANDOMIZE_INPUT(self) -> bool:
        return self._RANDOMIZE_INPUT

    @RANDOMIZE_INPUT.setter
    def RANDOMIZE_INPUT(self, randomize_input: bool) -> None:
        self._set_randomize_input(randomize_input)

    @change
    def _set_eta(self, eta: float) -> None:
        if self._ETA != eta:
            self._ETA = eta
            self.change(config_changed=True)

    @property
    def ETA(self) -> float:
        return self._ETA

    @ETA.setter
    def ETA(self, eta: float) -> None:
        self._set_eta(eta)

    @change
    def _set_blur_activated(self, blur_activated: bool) -> None:
        if self._BLUR_ACTIVATED != blur_activated:
            self._BLUR_ACTIVATED = blur_activated
            self.change(config_changed=True)

    @property
    def BLUR_ACTIVATED(self) -> bool:
        return self._BLUR_ACTIVATED

    @BLUR_ACTIVATED.setter
    def BLUR_ACTIVATED(self, blur_activated: bool) -> None:
        self._set_blur_activated(blur_activated)

    @change
    def _set_blur_kernel(self, blur_kernel: tuple) -> None:
        if self._BLUR_KERNEL != blur_kernel:
            self._BLUR_KERNEL = blur_kernel
            self.change(config_changed=True)

    @property
    def BLUR_KERNEL(self) -> tuple:
        return self._BLUR_KERNEL

    @BLUR_KERNEL.setter
    def BLUR_KERNEL(self, blur_kernel: tuple) -> None:
        self._set_blur_kernel(blur_kernel)

    @change
    def _set_blur_sigma(self, blur_sigma: int) -> None:
        if self._BLUR_SIGMA != blur_sigma:
            self._BLUR_SIGMA = blur_sigma
            self.change(config_changed=True)

    @property
    def BLUR_SIGMA(self) -> int:
        return self._BLUR_SIGMA

    @BLUR_SIGMA.setter
    def BLUR_SIGMA(self, blur_sigma: int) -> None:
        self._set_blur_sigma(blur_sigma)

    @change
    def _set_blur_frequency(self, blur_frequency: int) -> None:
        if self._BLUR_FREQUENCY != blur_frequency:
            self._BLUR_FREQUENCY = blur_frequency
            self.change(config_changed=True)

    @property
    def BLUR_FREQUENCY(self) -> int:
        return self._BLUR_FREQUENCY

    @BLUR_FREQUENCY.setter
    def BLUR_FREQUENCY(self, blur_frequency: int) -> None:
        self._set_blur_frequency(blur_frequency)

    @change
    def _set_l2_activated(self, l2_activated: bool) -> None:
        if self._L2_ACTIVATED != l2_activated:
            self._L2_ACTIVATED = l2_activated
            self.change(config_changed=True)

    @property
    def L2_ACTIVATED(self) -> bool:
        return self._L2_ACTIVATED

    @L2_ACTIVATED.setter
    def L2_ACTIVATED(self, l2_activated: bool) -> None:
        self._set_l2_activated(l2_activated)

    @change
    def _set_l2_lambda(self, L2_LAMBDA: float) -> None:
        if self._L2_LAMBDA != l2_lambda:
            self._L2_LAMBDA = l2_lambda
            self.change(config_changed=True)

    @property
    def L2_LAMBDA(self) -> float:
        return self._L2_LAMBDA

    @L2_LAMBDA.setter
    def L2_LAMBDA(self, l2_lambda: float) -> None:
        self._set_l2_lambda()

    @change
    def _set_norm_clipping_activated(self, activated: bool) -> None:
        if self._NORM_CLIPPING_ACTIVATED != activated:
            self._NORM_CLIPPING_ACTIVATED = activated
            self.change(config_changed=True)

    @property
    def NORM_CLIPPING_ACTIVATED(self) -> bool:
        return self._NORM_CLIPPING_ACTIVATED

    @NORM_CLIPPING_ACTIVATED.setter
    def NORM_CLIPPING_ACTIVATED(self, activated: bool) -> None:
        self._set_norm_clipping_activated(activated)

    @change
    def _set_norm_clipping_frequency(self, frequency: int) -> None:
        if self._NORM_CLIPPING_FREQUENCY != frequency:
            self._NORM_CLIPPING_FREQUENCY = frequency
            self.change(config_changed=True)

    @property
    def NORM_CLIPPING_FREQUENCY(self) -> int:
        return self._NORM_CLIPPING_FREQUENCY

    @NORM_CLIPPING_FREQUENCY.setter
    def NORM_CLIPPING_FREQUENCY(self, frequency: int) -> None:
        self._set_norm_clipping_frequency(frequency)

    @change
    def _set_norm_percentile(self, norm_percentile: float) -> None:
        if self._NORM_PERCENTILE != norm_percentile:
            self._NORM_PERCENTILE = norm_percentile
            self.change(config_changed=True)

    @property
    def NORM_PERCENTILE(self) -> float:
        return self._NORM_PERCENTILE

    @NORM_PERCENTILE.setter
    def NORM_PERCENTILE(self, norm_percentile: float) -> None:
        self._set_norm_percentile(norm_percentile)

    @change
    def _set_contribution_clipping_activated(self, activated: bool) -> None:
        if self._CONTRIBUTION_CLIPPING_ACTIVATED != activated:
            self._CONTRIBUTION_CLIPPING_ACTIVATED = activated
            self.change(config_changed=True)

    @property
    def CONTRIBUTION_CLIPPING_ACTIVATED(self) -> bool:
        return self._CONTRIBUTION_CLIPPING_ACTIVATED

    @CONTRIBUTION_CLIPPING_ACTIVATED.setter
    def CONTRIBUTION_CLIPPING_ACTIVATED(self, activated: bool) -> None:
        self._set_contribution_clipping_activated(activated)

    @change
    def _set_contribution_clipping_frequency(self, frequency: int) -> None:
        if self._CONTRIBUTION_CLIPPING_FREQUENCY != frequency:
            self._CONTRIBUTION_CLIPPING_FREQUENCY = frequency
            self.change(config_changed=True)

    @property
    def CONTRIBUTION_CLIPPING_FREQUENCY(self) -> int:
        return self._CONTRIBUTION_CLIPPING_FREQUENCY

    @CONTRIBUTION_CLIPPING_FREQUENCY.setter
    def CONTRIBUTION_CLIPPING_FREQUENCY(self, frequency: int) -> None:
        self._set_contribution_clipping_frequency(frequency)

    @change
    def _set_contribution_percentile(self, percentile: float) -> None:
        if self._CONTRIBUTION_PERCENTILE != percentile:
            self._CONTRIBUTION_PERCENTILE = percentile
            self.change(config_changed=True)

    @property
    def CONTRIBUTION_PERCENTILE(self) -> float:
        return self._CONTRIBUTION_PERCENTILE

    @CONTRIBUTION_PERCENTILE.setter
    def CONTRIBUTION_PERCENTILE(self, percentile: float) -> None:
        self._set_contribution_percentile(percentile)

    @change
    def _set_border_reg_activated(self, activated: bool) -> None:
        if self._BORDER_REG_ACTIVATED != activated:
            self._BORDER_REG_ACTIVATED = activated
            self.change(config_changed=True)

    @property
    def BORDER_REG_ACTIVATED(self) -> bool:
        return self._BORDER_REG_ACTIVATED

    @BORDER_REG_ACTIVATED.setter
    def BORDER_REG_ACTIVATED(self, activated: bool) -> None:
        self._set_border_reg_activated(activated)

    @change
    def _set_border_factor(self, border_factor: float) -> None:
        if self._BORDER_FACTOR != border_factor:
            self._BORDER_FACTOR = border_factor
            self.change(config_changed=True)

    @property
    def BORDER_FACTOR(self) -> float:
        return self._BORDER_FACTOR

    @BORDER_FACTOR.setter
    def BORDER_FACTOR(self, border_factor: float) -> None:
        self._set_border_factor(border_factor)

    @change
    def _set_border_exp(self, border_exp: float) -> None:
        if self._BORDER_EXP != border_exp:
            self._BORDER_EXP = border_exp
            self.change(config_changed=True)

    @property
    def BORDER_EXP(self) -> float:
        return self._BORDER_EXP

    @BORDER_EXP.setter
    def BORDER_EXP(self, border_exp: float) -> None:
        self._set_border_exp(border_exp)

    @change
    def _set_larger_image(self, activated: bool) -> None:
        if self._LARGER_IMAGE != activated:
            self._LARGER_IMAGE = activated
            self.change(config_changed=True)

    @property
    def LARGER_IMAGE(self) -> bool:
        return self._LARGER_IMAGE

    @LARGER_IMAGE.setter
    def LARGER_IMAGE(self, activated: bool) -> None:
        self._set_larger_image(activated)

    @change
    def _set_image_size_x(self, image_size_x: int) -> None:
        if self._IMAGE_SIZE_X != image_size_x:
            self._IMAGE_SIZE_X = image_size_x
            self.change(config_changed=True)

    @property
    def IMAGE_SIZE_X(self) -> int:
        return self._IMAGE_SIZE_X

    @IMAGE_SIZE_X.setter
    def IMAGE_SIZE_X(self, image_size_x: int) -> None:
        self._set_image_size_x(image_size_x)

    @change
    def _set_image_size_y(self, image_size_y: int) -> None:
        if self._IMAGE_SIZE_Y != image_size_y:
            self._IMAGE_SIZE_Y = image_size_y
            self.change(config_changed=True)

    @property
    def IMAGE_SIZE_Y(self) -> int:
        return self._IMAGE_SIZE_Y

    @IMAGE_SIZE_Y.setter
    def IMAGE_SIZE_Y(self, image_size_y: int) -> None:
        self._set_image_size_y(image_size_y)

    @change
    def _set_jitter(self, activated: bool) -> None:
        if self._JITTER != activated:
            self._JITTER = activated
            self.change(config_changed=True)

    @property
    def JITTER(self) -> bool:
        return self._JITTER

    @JITTER.setter
    def JITTER(self, activated: bool) -> None:
        self._set_jitter(activated)

    @change
    def _set_jitter_strength(self, jitter_strength: int) -> None:
        if self._JITTER_STRENGTH != jitter_strength:
            self._JITTER_STRENGTH = jitter_strength
            self.change(config_changed=True)

    @property
    def JITTER_STRENGTH(self) -> int:
        return self._JITTER_STRENGTH

    @JITTER_STRENGTH.setter
    def JITTER_STRENGTH(self, jitter_strength: int) -> None:
        self._set_jitter_strength(jitter_strength)

    @change
    def _set_wrap_around(self, activated: bool) -> None:
        if self._WRAP_AROUND != activated:
            self._WRAP_AROUND = activated
            self.change(config_changed=True)

    @property
    def WRAP_AROUND(self) -> bool:
        return self._WRAP_AROUND

    @WRAP_AROUND.setter
    def WRAP_AROUND(self, activated: bool) -> None:
        self._set_wrap_around(activated)

    @change
    def _set_loss_goal(self, loss_goal: float) -> None:
        if self._LOSS_GOAL != loss_goal:
            self._LOSS_GOAL = loss_goal
            self.change(config_changed=True)

    @property
    def LOSS_GOAL(self) -> float:
        return self._LOSS_GOAL

    @LOSS_GOAL.setter
    def LOSS_GOAL(self, loss_goal: float) -> None:
        self._set_loss_goal(loss_goal)

    @change
    def _set_loss_count(self, loss_count: int) -> None:
        if self._LOSS_COUNT != loss_count:
            self._LOSS_COUNT = loss_count
            self.change(config_changed=True)

    @property
    def LOSS_COUNT(self) -> int:
        return self._LOSS_COUNT

    @LOSS_COUNT.setter
    def LOSS_COUNT(self, loss_count: int) -> None:
        self._set_loss_count(loss_count)


    @change
    def _set_max_steps(self, max_steps: int) -> None:
        if self._MAX_STEPS != max_steps:
            self._MAX_STEPS = max_steps 
            self.change(config_changed=True)

    @property
    def MAX_STEPS(self) -> int:
        return self._MAX_STEPS

    @MAX_STEPS.setter
    def MAX_STEPS(self, max_steps: int) -> None:
        self._set_max_steps(max_steps)

    @change
    def _set_tensorboard_activated(self, tensorboard_activated: bool) -> None:
        if self._TENSORBOARD_ACTIVATED != tensorboard_activated:
            self._TENSORBOARD_ACTIVATED = tensorboard_activated
            self.change(config_changed=True)

    @property
    def TENSORBOARD_ACTIVATED(self) -> bool:
        return self._TENSORBOARD_ACTIVATED

    @TENSORBOARD_ACTIVATED.setter
    def TENSORBOARD_ACTIVATED(self, tensorboard_activated: bool) -> None:
        self._set_tensorboard_activated(tensorboard_activated)

    @change
    def _set_normalize_output(self, normalize_output: bool) -> None:
        if self._NORMALIZE_OUTPUT != normalize_output:
            self._NORMALIZE_OUTPUT = normalize_output
            self.change(config_changed=True)

    @property
    def NORMALIZE_OUTPUT(self) -> bool:
        return self._NORMALIZE_OUTPUT

    @NORMALIZE_OUTPUT.setter
    def NORMALIZE_OUTPUT(self, normalize_output: bool) -> None:
        self._set_normalize_output(normalize_output)

    def copy(self):
        return copy.copy(self)
