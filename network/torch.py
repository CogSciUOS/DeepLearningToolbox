
# Some torch remarks:
# * x.cpu() and x.cuda() is the old (pre 0.4) and now deprecated way
#   for assigning computations to devices. The modern (now recommended)
#   way is to say x.to('cpu') or x.to('cuda:0')
#
# Questions:
# * what does torch.cuda.synchronize() do?

# standard imports
from __future__ import absolute_import  # FIXME[question]: what is this?
from typing import Tuple
import logging

import importlib.util

# third party imports
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

# toolbox imports
from . import Network as BaseNetwork

# logging
LOG = logging.getLogger(__name__)


# FIXME[todo]: check docstrings
# FIXME[todo]: not tested yet!
# FIXME[todo]: need to clean up

# FIXME[todo]: cuda activation (if available)


class Network(BaseNetwork):
    """A class implmenting the network interface (BaseNetwork) to access
    feed forward networks implemented in (py)Torch.  This is
    essentially a wrapper around a :py:class:`torch.nn.Module`.

    Some general remarks on (py)torch:

    * Torch convolution follows the channel first scheme, that is
      the shape of a 2D convolution is (batch, channel, height, width).

    * Torch does not store activation values. However, one can
      register hooks to be executed before or after the forward or
      backward propagation.  These hooks can be use to access and
      store input and output values.

    * The nn.Module (layers) does not have a name. I store the
      key by which they have been registered in the parent Module
      under the property _name. These names will also function
      as layer_id to identify individual layers from the outside.


    Properties
    ----------

    torch_module: :py:class:`torch.nn.Module`

    Arguments
    ---------
    cls: Union[type, str, Tuple[str, str]]
        Either a sublcass of :py:class:`torch.nn.Module` or a str
        containing a fully qualified class name (module + class) or
        a pair consisting of the name of a python file (`.py`) and
        a name of a class defined in that file.
    """

    # _model: the underlying torch.nn.Module
    _model: nn.Module

    # _input_shapes: a dictionary mapping layer names to input_shapes
    _input_shapes: dict = None
    _output_shapes: dict = None
    _hooks: dict = {}

    # _python_class: a sublcass of :py:class:`torch.nn.Module`
    _python_class: type = None
    # _torch_filename: name of the weights file (either .pth or .pth.tar)
    _torch_filename: str = None

    # _use_cude: use cuda if available
    _use_cuda: bool = False

    # _train_mode: run model in train mode (True) or eval mode (False)
    _train_mode: bool = False

    def __init__(self, *args, cuda: bool = True, train: bool = False,
                 input_shape: Tuple[int, int] = None,
                 **kwargs):
        """
        Load Torch model.

        Parameters
        ----------
        model_file
            Path to the .h5 model file.
        """
        super().__init__(**kwargs)

        #
        # Try to get a model:
        #

        if len(args) > 0:

            if isinstance(args[0], nn.Module):
                self._model = args[0]
            elif isinstance(args[0], str) and args[0].endswith(".pth.tar"):
                self._torch_filename = args[0]
            elif (isinstance(args[0], Tuple) and len(args[0] == 2) and
                  args[0][0].endswith(".py")):
                spec = importlib.util.spec_from_file_location('torchnet',
                                                              args[0][0])
                net_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(net_module)
                net_class_name = kwargs.get('network_class', 'Net')
                net_class = getattr(net_module, net_class_name)
                self._python_class = net_class()
            else:
                raise ValueError("Invalid arguments for "
                                 "constructing a TorchNetwork")

        if cuda and not torch.cuda.is_available():
            LOG.warning("torch CUDA is not available, "
                        "resorting to GPU computation")
            cuda = False
        self._use_cuda = cuda
        self._train_mode = train
        self._input_shape = input_shape

    def _prepare(self) -> None:

        if self._model is None:
            if not self._torch_filename:
                raise ValueError("No filename provided for loading "
                                 "torch model")

            if self._torch_filename.endswith(".pth.tar"):
                self._model = torch.load(self._torch_filename)
            elif (self._torch_filename.endswith(".pth") and
                  self._python_class is not None):
                self._model = self._python_class()
                self._model.load_state_dict(torch.load(self._torch_filename))
            else:
                raise ValueError("Invalid filene '{self._torch_filename}' "
                                 "for loading torch model.")

        # activate cuda (gpu support) if available
        if self._use_cuda:
            self._model.to('cuda')  # 'cuda:0'

        # set train/eval mode
        if self._train_mode:
            self._model.train()
        else:
            self._model.eval()

        # compute input shapes for all layers
        if self._input_shape is not None:
            self._compute_layer_shapes(self._input_shape)

    def _compute_layer_shapes(self, input_shape: tuple) -> None:
        """Compute the input and output shapes of all layers.
        The shapes are determined by propagating some dummy input through
        the network.

        input_shape:
            The shape of an input sample. May or may not include batch (B)
            or channel (C) dimension. If so, channel should be last, i.e.
            (N,H,W,C)
        """

        input_shape = self._canonical_input_shape(input_shape)
        # Torch convolution follows the channel first scheme, that is
        # the shape of a 2D convolution is (batch, channel, height, width).
        torch_input_shape = tuple(input_shape[_] for _ in [0, 3, 1, 2])

        self._input_shapes = {}
        self._output_shapes = {}
        self._prepare_hooks(self._shape_hook)
        self._model(Variable(torch.zeros(*torch_input_shape), volatile=True))
        self._remove_hooks()

    def _get_number_of_input_channels(self) -> int:
        """Get the number of input channels for this network.
        This is the number of channels each input given to the network
        should have.  Usually this coincides with the number of
        channels in the first layer of the network.

        Returns
        -------
        int
            The number of input channels or 0 if the network does not
            have input channels.
        """
        first = self._get_first_layer()
        return first.in_channels if self.layer_is_convolutional(first) else 0

    def _prepare_hooks(self, hook, layer_ids=None) -> None:

        if layer_ids is None:
            layer_ids = self.layer_ids

        for id in layer_ids:
            module = self._model._modules[id]
            module._name = id  # FIXME[hack]: how to acces the module name?
            self._hooks[id] = module.register_forward_hook(hook)

    def _remove_hooks(self, layer_ids=None) -> None:
        if layer_ids is None:
            layer_ids = list(self._hooks.keys())
        for id in layer_ids:
            self._hooks[id].remove()
            del self._hooks[id]

    def _shape_hook(self, module, input, output):
        name = module._name  # FIXME[hack]: how to acces the module name?
        # input[0].size() will be (N, C, H, W) -> store (H ,W, C)
        input_shape = input[0].size()
        self._input_shapes[name] = (*input_shape[2:], input_shape[1])
        # output.size() will be (N, C, H, W) -> store (C, H ,W)
        output_shape = output.size()
        self._output_shapes[name] = (*output_shape[2:], output_shape[1])

    def _activation_hook(self, module, input, output):
        name = module._name  # FIXME[hack]: how to acces the module name?
        # output is a torch.autograd.variable.Variable,
        # output.data holds the actual data
        # FIXME[quesion]: it is said in the documentation, that
        # the TorchTensor and the numpy array share the same memory.
        # However, the TorchTensor may be removed after usage (is this true?)
        # what will then happen with the numpy array? do we need to
        # copy or is that a waste of resources?
        self._activations[name] = output.data.numpy().copy()
        if len(output.data.size()) == 4:  # convolution, (N,C,H,W)
            self._activations[name] = \
                self._activations[name].transpose(0, 2, 3, 1)

    def _get_layer(self, layer_id) -> nn.Module:
        """Get a torch Module representing the layer for the given identifier.

        Parameters
        ----------
        layer_id:
            Identifier of a layer in this network.

        Returns
        -------
        The layer for the given identifier.
        """
        return (layer_id if isinstance(layer_id, torch.nn.Module)
                else self._model._modules[layer_id])

    def _get_first_layer(self) -> nn.Module:
        """Get a torch Module representing the first layer of this network.

        Returns
        -------
        nn.Module
            The first layer of this network.
        """
        return self._get_layer(self.layer_ids[0])

    @property
    def layer_ids(self) -> list:
        """Get list of layer ids. These ids can be used to access layers via
        this Network API.

        Returns
        -------
        The list of layer identifiers.
        """
        return list(self._model._modules.keys())

    def layer_is_convolutional(self, layer_id) -> bool:
        """Check if the given layer is a convolutional layer. If so,
        additional information can be obtained by the methods
        get_layer_kernel_size, get_layer_input_channels,
        get_layer_output_channels, get_layer_stride,
        get_layer_padding, get_layer_output_padding, and
        get_layer_dilation.

        Parameters
        ----------
        layer:
            Identifier of a layer in this network.

        Returns
        -------
        bool
            True for convolutional layers, else False.

        """
        return isinstance(self._get_layer(layer_id), nn.modules.conv.Conv2d)

    def get_layer_kernel_size(self, layer_id) -> int:
        """The size of the kernel in a cross-correlation/convolution
        operation. This is just the spatial extension and does not
        include the number of channels.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        layer = self._get_layer(layer_id)
        self._check_layer_is_convolutional(layer)
        return layer.kernel_size

    def get_layer_input_channels(self, layer_id) -> int:
        """The number of input channels for a cross-correlation/convolution
        operation.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        layer = self._get_layer(layer_id)
        self._check_layer_is_convolutional(layer)
        return layer.in_channels

    def get_layer_output_channels(self, layer_id) -> int:
        """The number of output channels for a cross-correlation/convolution
        operation.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        layer = self._get_layer(layer_id)
        self._check_layer_is_convolutional(layer)
        return layer.out_channels

    def get_layer_stride(self, layer_id) -> (int, int):
        """The stride for the cross-correlation/convolution operation.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.

        """
        layer = self._get_layer(layer_id)
        self._check_layer_is_convolutional(layer)
        return layer.stride

    def get_layer_padding(self, layer_id) -> (int,int):
        """The padding for the cross-correlation/convolution operation, i.e,
        the number of rows/columns (on both sides) by which the input
        is extended (padded with zeros) before the operation is
        applied.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        layer = self._get_layer(layer_id)
        self._check_layer_is_convolutional(layer)
        return layer.padding

    def get_layer_output_padding(self, layer_id) -> (int,int):
        """The output padding for the cross-correlation/convolution operation.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        layer = self._get_layer(layer_id)
        self._check_layer_is_convolutional(layer)
        return layer.output_padding

    def get_layer_dilation(self, layer_id) -> (int, int):
        """The dilation for the cross-correlation/convolution operation, i.e,
        the horizontal/vertical offset between adjacent filter
        rows/columns.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        layer = self._get_layer(layer_id)
        self._check_layer_is_convolutional(layer)
        return layer.dilation

    def get_layer_input_shape(self, layer_id) -> tuple:
        """
        Give the shape of the input of the given layer.

        Parameters
        ----------
        layer_id

        Returns
        -------
        (units) for dense layers
        (height, width, channels) for convolutional layers

        Raises
        ------
        RuntimeError
            If the network shape is not determine yet.
        """
        if self._input_shapes is None:
            raise RuntimeError("Network shapes have not been fixed yet.")
        return self._input_shapes[layer_id]

    def get_layer_output_shape(self, layer_id) -> tuple:
        """
        Give the shape of the output of the given layer.

        Parameters
        ----------
        layer_id

        Returns
        -------
        (units) for dense layers
        (height, width, channels) for convolutional layers

        Raises
        ------
        RuntimeError
            If the network shape is not determine yet.
        """
        if self._output_shapes is None:
            raise RuntimeError("Network shapes have not been fixed yet.")
        return self._output_shapes[layer_id]

    def get_layer_weights(self, layer_id) -> np.ndarray:
        """
        Returns weights INCOMING to the
        layer of the model
        shape of the weights variable should be
        coherent with the get_layer_output_shape function.

        Parameters
        ----------
        layer_id :
             An identifier for a layer.

        Returns
        -------
        ndarray
            Weights of the layer. For convolutional layers this will
            be (H,W,C_in,C_out)

        """
        layer = self._get_layer(layer_id)
        weights = layer.weight.data.numpy()
        if self.layer_is_convolutional(layer):
            weights = weights.transpose(2, 3, 0, 1)
        return weights

    def get_layer_biases(self, layer_id) -> np.ndarray:
        """
        Returns weights INCOMING to the
        layer of the model
        shape of the weights variable should be
        coherent with the get_layer_output_shape function.

        Parameters
        ----------
        layer_id :
             An identifier for a layer.

        Returns
        -------
        ndarray
            Bias values for the layer. For convolutional layers this will
            be one bias value per (output) channel.
        """
        layer = self._get_layer(layer_id)
        biases = layer.bias.data.numpy()
        return biases

    def get_activations(self, layer_ids, input_samples: np.ndarray) -> list:
        """Gives activations values of the network/model
        for a given layername and an input (inputsample).

        Parameters
        ----------
        layer_ids: The layer(s) the activations should be fetched for,
            either an atomic id or a list of ids.
        input_samples:
            Array of samples the activations should be computed for.
            Default format is 4D: (batch,width,height,channel)
            If batch or channel is 1, it can be omitted.

        Returns
        -------
        np.ndarray

        """
        # We need to know the network input shape to get a cannonical
        # representation of the input_samples.
        if self._input_shapes is None or self._output_shapes is None:
            self._compute_layer_shapes(input_samples.shape)

        _layer_ids, _input_samples = \
            super().get_activations(layer_ids, input_samples)

        # pytorch expects channel first (N,C,H,W)
        # FIXME[concept]: we may want to directly get the preferred order
        # from the _canonical_input_data method!
        _input_samples = _input_samples.transpose((0, 3, 1, 2))

        torch_samples = torch.from_numpy(_input_samples)
        torch_input = Variable(torch_samples, volatile=True)

        # FIXME[todo]: use GPU
        # if self._use_cuda:
        #    torch_input = torch_input.cuda()

        # prepare to record the activations
        self._activations = {}
        self._prepare_hooks(self._activation_hook, _layer_ids)

        torch_output = self._model(torch_input)

        self._remove_hooks(layer_ids)

        # if no batch data was provided, remove batch dimension from
        # activations
        if (_input_samples.shape[0] == 1
            and len(input_samples.shape) < 4
            and input_samples.shape[0] != 1):
            for id in self._activations.keys():
                self._activations[id] = self._activations[id].squeeze(0)

        # if a single layer_id was given (not a list), then just return
        # a single activtion array
        return ([self._activations[_] for _ in layer_ids]
                if isinstance(layer_ids, list)
                else self._activations[layer_ids])

from typing import Union
from torchvision import transforms
from .network import Classifier
from dltb.util.image import imread
from PIL import Image
from dltb.tool.classifier import ImageClassifier


class DemoResnetNetwork(ImageClassifier, Classifier, Network):
    """An experimental Torch Network based on ResNet.

    https://pytorch.org/hub/pytorch_vision_resnet/
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(scheme='ImageNet', lookup='torch', **kwargs)

    def _prepare(self) -> None:

        self._model = torch.hub.load('pytorch/vision:v0.6.0',
                                     'resnet18', pretrained=True)
        # or any of these variants
        # model = torch.hub.load('pytorch/vision:v0.6.0',
        #                        'resnet34', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0',
        #                        'resnet50', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0',
        #                        'resnet101', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0',
        #                        'resnet152', pretrained=True)
        self._scheme.prepare()
        super()._prepare()

        self._preprocess_image = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _image_as_batch(self, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            # image = imread(image)
            image = Image.open(image)

        # image should be PIL Image. Got <class 'numpy.ndarray'>
        image_tensor = self._preprocess_image(image)

        # create a mini-batch as expected by the model
        image_batch = image_tensor.unsqueeze(0)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            image_batch = image_batch.to('cuda')

        return image_batch

    #
    # Implementation of the SoftClassifier interface
    #

    def class_scores(self, inputs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self._model(inputs)

        # Tensor of shape 1000, with confidence scores over Imagenet's
        # 1000 classes
        # print(output[0])

        # The output has unnormalized scores. To get probabilities,
        # you can run a softmax on it.
        outputs = torch.nn.functional.softmax(logits, dim=1)

        if torch.cuda.is_available():
            outputs = outputs.cpu()

        outputs = outputs.numpy()
        return outputs
