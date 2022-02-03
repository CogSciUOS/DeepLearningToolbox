"""This module provides some convenience code to integrate some parts of
the torchvision package into the Deep Learning Toolbox.


import torchvision.models as models

alexnet = models.alexnet(pretrained=True)

for name, module in alexnet.named_modules():
    print(name)


"""

# standard imports
from typing import Iterable
from types import FunctionType

# thirdparty imports
import torch
import torchvision.models

# toolbox imports
from dltb.thirdparty.datasource.imagenet import ImagenetClassifier
from .network import ImageNetwork, ClassifierNetwork


class Network(ImageNetwork, ClassifierNetwork, ImagenetClassifier):
    """Convenience class to access torchvision models in the Deep Learning
    Toolbox.
    """

    @staticmethod
    def pretrained() -> Iterable[str]:
        """The pretrained models available in torchvision.

        """
        for name, model in torchvision.models.__dict__.items():
            if isinstance(model, FunctionType):
                yield name

    @staticmethod
    def framework_info():
        """Provide information on the underlying framework running
        the model.
        """
        return (f"Torch version {torch.__version__} "
                f"(torchvision {torchvision.__version__})")

    def __init__(self, model=None, lookup: str = 'torch', **kwargs):
        super().__init__(model=model, lookup=lookup, **kwargs)

    def _prepare(self) -> None:
        if self._model is None:
            if not isinstance(self._model_init, str):
                raise TypeError("Illegal type for initializing a torchvision "
                                f"network model: {type(self._model_init)}")
            if isinstance(getattr(torchvision.models, self._model_init, None),
                          FunctionType):
                model_class = getattr(torchvision.models, self._model_init)

                # download and load pretrained inceptionv3 model
                #
                # The flag 'transform_input' indicates, that the input
                # is preprocessed according to the method with which
                # the model was trained on ImageNet (this includes
                # centering and normalization, but not other
                # operations like resizing or scaling to the interval
                # [0,1]). If set to False, you should manually
                # preprocess input data before handing it to the
                # network.  The default is False.
                self._model = model_class(pretrained=True)
            else:
                raise ValueError(f"No model with name '{model}' found in "
                                 "the Torchvision model zoo!")
        super()._prepare()


class AlexNet(Network):
    """The legendary AlexNet.
    """

    def __init__(self, model='alexnet', **kwargs) -> None:
        super().__init__(model=model, **kwargs)
