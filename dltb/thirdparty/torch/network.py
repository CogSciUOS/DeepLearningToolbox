"""Torch implementation of the Deep Learning Toolbox Network API.
"""
# Some torch remarks:
# * x.cpu() and x.cuda() is the old (pre 0.4) and now deprecated way
#   for assigning computations to devices. The modern (now recommended)
#   way is to say x.to('cpu') or x.to('cuda:0')
#
# * There is a simple functional API allowing to apply a network or layer
#   (torch.nn.Module) to same data (Tensor): module(data)
#   - Tensor vs. Variable:
#     for Pytorch prior to 0.4.0, Variable and Tensor were two different
#     things. With Pytorch 0.4.0 Variable and Tensor have been merged.
#   - volatile is deprecated:
#     for Pytorch prior to 0.4.0 the volatile=True flag could be used
#     to signal that all intermediate results can be dropped
#
# * Some transformation functions seem to rely on thirdparty libraries.
#   For example image transformations, like torchvision.transforms.Resize(),
#   rely on PIL.Image (pillow library).
#
# Questions:
# Q: what is a torch.autograd.Variable?
# Q: what does torch.cuda.synchronize() do?
#
# Q: how to acces the module name?
#    [name for name in dir(module) if 'name' in name] evaluates to
#    ['_get_name', '_named_members', '_tracing_name', 'named_buffers',
#      'named_children', 'named_modules', 'named_parameters']
#    name = module._get_name()
# A: it seems that not modules have not name. Only submodules can
#    be registered with a name ...


# standard imports
from typing import Tuple, Union, List, Iterable, Optional
from collections import OrderedDict
import logging
import importlib.util

# third party imports
import numpy as np
import PIL
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

# toolbox imports
from dltb.base.data import Datalike
from dltb.base.image import Image, Imagelike
from dltb.tool.classifier import ClassIdentifier
from dltb.network import Network as BaseNetwork
from dltb.network import Classifier as BaseClassifier
from dltb.network import ImageNetwork as BaseImageNetwork
from dltb.network import layer as Base
from dltb.util.array import DATA_FORMAT_CHANNELS_FIRST
from .util import Debug

# logging
LOG = logging.getLogger(__name__)


# FIXME[todo]: check docstrings
# FIXME[todo]: not tested yet!
# FIXME[todo]: need to clean up

# FIXME[todo]: cuda activation (if available)


class Layer(Base.Layer):
    """A torch :py:class:`Layer` encapsulates a
    :py:class:`torch.nn.Module` that should be considered as
    one layer.

    _module: nn.Module
        The torch neural network module realizing this layer.

    _input_shape: Tuple
        A tuple holding the expected input shape for this layer.
        this is ordered (batch, ..., channels).
        The batch size is not included, undetermined values
        are set to None.

    _output_shape: Tuple
        A tuple holding the expected input shape for this layer.
        this is ordered (batch, ..., channels).
        The batch size is not included, undetermined values
        are set to None.
    """

    def __new__(cls, module: nn.Module, **kwargs) -> None:
        LOG.info("torch: new Layer with type %s", type(module))
        if isinstance(module, nn.Conv2d):
            real_cls = Conv2D
        elif isinstance(module, nn.Sequential):
            real_cls = Sequential
        else:
            real_cls = cls

        if super().__new__ is object.__new__:
            # object.__new__() takes exactly one argument
            return object.__new__(real_cls)
        return super().__new__(real_cls, **kwargs)

    def __init__(self, module: nn.Module, **kwargs) -> None:
        super().__init__(**kwargs)
        self._module = module

        # FIXME[todo]: Estimate the input shape
        first_shape = None
        last_shape = None
        # print(f"{type(module)}:")
        for parameter in module.parameters():
            # print('  -', parameter.shape)
            if first_shape is None:
                first_shape = parameter.shape
            last_shape = parameter.shape
        if first_shape is not None:
            self._input_shape = \
                ((None, first_shape[1]) + (None,) * (len(first_shape)-2)
                 if len(first_shape) > 1 else (None, first_shape[0]))
            self._output_shape = \
                (None, last_shape[0]) + (None,) * (len(last_shape)-2)
        else:
            self._input_shape = None
            self._output_shape = None

        LOG.debug("Created new torch layer: %s (%s): input=%s, output=%s",
                  type(self), type(module),
                  self.input_shape, self.output_shape)

    @property
    def module(self) -> nn.Module:
        """The torch neural network module realizing this layer.
        """
        return self._module

    def adapt_shapes(self, module: nn.Module,
                     incoming: torch.Tensor, outgoing: torch.Tensor) -> None:
        # pylint: disable=unused-argument
        """Adapt the input and output shape of this Layer to reflect
        the size of the given sample data.
        """
        input_shape = incoming[0].size()
        output_shape = outgoing.size()
        self._input_shape = (None, *input_shape[1:])
        self._output_shape = (None, *output_shape[1:])

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """The shape of the tensor that is input to the layer,
        including the batch axis.
        The channel will be channel last.
        """
        return self._input_shape

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """The shape of the tensor that is output to the layer,
        including the batch axis.
        The channel will be channel last.
        """
        return self._output_shape


class Sequential(Layer):
    """A layer based on a torch :py:class:`nn.Sequential` module.
    """

    def _first_module_with_attr(self, attr: str):  # -> nn.Module:
        """Provide the first operation providing the requested attribute.
        """
        for module in self._module:
            if hasattr(module, attr):
                return module
        raise ValueError("Layer contains no operation with"
                         f"attribute '{attr}'")

    def receptive_field(self, point1: Tuple[int, ...],
                        point2: Tuple[int, ...] = None
                        ) -> (Tuple[int, ...], Tuple[int, ...]):
        """The receptive field of a layer.

        Parameters
        ----------
        point1:
            The upper left corner of the region of interest.
        point2:
            The lower right corner of the region of interest.
            If none is given, the region is assumed to consist of just
            one point.

        Returns
        -------
        point1, point2:
            The upper left corner and the lower right corner of the
            receptive field for the region of interest.
        """


# -----------------------------------------------------------------------------


class NeuralLayer(Layer, Base.NeuralLayer):
    """Torch implentation of the :py:class:`Base.NeuralLayer`
    interface.
    """

    @property
    def parameters(self):
        return (self.weights, self.bias)

    @property
    def num_parameters(self):
        return self.weights.size + self.bias.size

    @property
    def _weights(self) -> np.ndarray:
        return self._module.weight

    @property
    def _bias(self) -> np.ndarray:
        return self._module.bias

    @property
    def weights(self) -> np.ndarray:
        return self._weights.numpy()

    @property
    def bias(self) -> np.ndarray:
        return self._bias.numpy()


class StridingLayer(Layer, Base.StridingLayer):
    """Torch implentation of the :py:class:`Base.StridingLayer`
    interface.
    """

    @property
    def filter_size(self):
        return self._module.kernel_size

    @property
    def strides(self) -> Tuple[int]:
        return self._module.stride

    @property
    def padding(self) -> Tuple[int]:
        """The amount of padding to be added on both sides
        along each axis.
        """
        return self._module.padding

    @property
    def activation_tensor(self):
        """The tensor that contains the activations of the layer."""
        # For now assume that the last operation is the activation.
        # Maybe differentiate with subclasses later.
        # return self._ops[0].outputs[0]
        # FIXME[hack]: this does not really seem into the concept
        # of a torch layer (i.e., a torch.nn.Module)
        return None


class Dense(NeuralLayer, Base.Dense):
    """Torch implentation of the :py:class:`Base.Dense`
    interface.
    """


class Conv2D(NeuralLayer, StridingLayer, Base.Conv2D):
    """Torch implentation of the :py:class:`Base.Conv2D`
    interface.
    """

    @property
    def kernel_size(self) -> Tuple[int, int]:
        return self._module.kernel_size

    @property
    def dilation(self) -> Tuple[int, int]:
        """The diation describes the stepsize in which this filter is
        applied. For example, a dilation of `(2, 2)` means that the
        kernel is only applied at every second horizontal and vertical
        position, that is at every fourth position in total. Hence the
        output will have half the with and height of the input.

        """
        print(f"torch.Conv2D: dilation={self._module.dilation}")
        return self._module.dilation

    @property
    def filters(self):
        return self.output_shape[1]

    @property
    def weight_tensor(self):
        """The weight (parameter) tensor of this :py:class:`Layer`.
        """
        # The last input of the first operation with strides
        # should correspond to the (first part of the) weights.
        # FIXME[old]: is this old?
        # striding_op = self._first_module_with_attr('strides')
        # return striding_op.inputs[-1]
        return self._module.weight


class MaxPooling2D(StridingLayer, Base.MaxPooling2D):
    """Torch implentation of the :py:class:`Base.MaxPooling2D`
    interface.
    """

    @property
    def pool_size(self) -> Tuple[int, int]:
        return self._module.kernel_size

    @property
    def dilation(self) -> Tuple[int, int]:
        """The diation describes the stepsize in which this filter is
        applied. For example, a dilation of `(2, 2)` means that the
        kernel is only applied at every second horizontal and vertical
        position, that is at every fourth position in total. Hence the
        output will have half the with and height of the input.

        """
        return self._module.dilation


class Dropout(Layer, Base.Dropout):
    """Torch implentation of the :py:class:`Base.Dropout`
    interface.
    """

    @property
    def rate(self) -> float:
        """The dropout rate indicating the ratio of inputs to be dropped.
        This should be a value between `0.0` and `1.0`
        """
        return self._module.p


class Flatten(Layer, Base.Flatten):
    """Torch implentation of the :py:class:`Base.Flatten`
    interface.
    """


# -----------------------------------------------------------------------------

Modellike = Union[torch.nn.Module, str]

class Network(BaseNetwork):
    # pylint: disable=too-many-instance-attributes
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

    _input_shape:

    _input_shapes:

    _output_shapes:

    _activations:

    _train_mode: bool

    Arguments
    ---------
    model: Union[type, str, Tuple[str, str]]
        Either a sublcass of :py:class:`torch.nn.Module` or a str
        containing a fully qualified class name (module + class) or
        a pair consisting of the name of a python file (`.py`) and
        a name of a class defined in that file.
    """

    # pytorch expects channel first (batch,channel,height,width)
    _internal_format = DATA_FORMAT_CHANNELS_FIRST

    # _model: the underlying torch.nn.Module
    _model: nn.Module

    # _hooks: a dictionary mapping layer names to hooks
    _hooks: dict = {}

    # _device: the device to perform computations on
    #   ('cpu', 'cuda', or 'cuda0')
    _device: torch.device = None

    # _train_mode: run model in train mode (True) or eval mode (False)
    _train_mode: bool = False

    # FIXME[todo]: move to some more suitable place ...
    @staticmethod
    def gpu_is_available():
        """Check if a GPU is available for running this model.
        """
        return torch.cuda.is_available()

    def __init__(self, model: Optional[Modellike] = None,
                 cuda: bool = True, train: bool = False,
                 input_shape: Tuple[int, int] = None,
                 **kwargs):
        """
        Load Torch model.

        Parameters
        ----------
        model_file
            Path to the .h5 model file.
        """
        LOG.info("torch.Network(%s, cuda=%s, train=%s, input_shape=%s)",
                 model if isinstance(model, str) else type(model),
                 cuda, train, input_shape)
        super().__init__(**kwargs)

        #
        # Initialize properties
        #
        self._input_shapes = None
        self._output_shapes = None
        self._activations = None
        self._train_mode = train
        self._input_shape = input_shape

        if isinstance(model, nn.Module):
            self._model, self._model_init = model, None
        else:
            self._model, self._model_init = None, model

        if cuda and not torch.cuda.is_available():
            LOG.warning("torch CUDA is not available, "
                        "resorting to GPU computation")
        if cuda:
            self._device = torch.device('cuda')  # 'cuda:0'
        else:
            self._device = torch.device('cpu')

    def _prepared(self) -> bool:
        return self._model is not None and super()._prepared()

    def _prepare(self) -> None:
        LOG.info("torch.Network: prepare: model=%s, model_init=%s",
                 type(self._model), self._model_init)

        #
        # Try to get a model:
        #
        if self._model is None and self._model_init is None:
            raise ValueError("No model provided for torch Network.")

        if self._model is not None:
            pass  # no model initialization is necessary
        elif (isinstance(self._model_init, str) and
                self._model_init.endswith(".pth.tar")):
            # load model from .pth.tar file
            self._model = torch.load(self._model)
        elif (isinstance(self._model_init, Tuple) and len(self._model == 2) and
              self._model[0].endswith(".py") and
              self._model[1].endswith(".pth")):
            # pair (module_name, weights_filename)
            # module_name: a python module defining a class 'Net'
            # weights_filename: name of a torch state dictionary
            module_name, weights_filename = self._model_init
            model_class_name = 'Net'
            module_spec = \
                importlib.util.spec_from_file_location('torchnet', module_name)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module_name)
            model_class = getattr(module, model_class_name)
            self._model = model_class()
            self._model.load_state_dict(torch.load(weights_filename))
        elif self._model is None:
            raise ValueError("Invalid arguments for "
                             "constructing a torch Network")

        if not isinstance(self._model, nn.Module):
            raise ValueError("Preparation of torch Network yielded bad type: "
                             f"{type(self._model)}")

        # assign the model to a device
        # (should be done before preparing the layers with _prepare_layers,
        # as this will propagate data through the model).
        self._model.to(self._device)

        super()._prepare()

        # set train/eval mode
        #
        # Some layers have different behavior during training and
        # evaluation.  By default all the modules are initialized to
        # train mode (model.train() setting model.training = True). To
        # switch to evaluation mode call model.eval().
        self._model.eval()
        if self._train_mode:
            self._model.train()
        else:
            self._model.eval()

    def _prepare_layers(self) -> None:
        super()._prepare_layers()

        # compute input shapes for all layers
        if self._input_shape is not None:
            self._compute_layer_shapes(self._input_shape)
        else:
            # FIXME[todo]: think of a general strategy to compute layer
            # sizes (may be done backward from the last dense layer)
            # In case of a fully convolutional network, there is no
            # fixed layer shape.
            NotImplementedError("Automatic determination of input shapes "
                                "for torch models is not implemented yet.")

    def _create_layer_dict(self) -> OrderedDict:
        """Create the mapping from layer ids to layer objects.

        Returns
        -------

        """
        layer_dict = {}
        self._add_to_layer_dict(layer_dict, self._model, recurse=True)
        return layer_dict

    def _add_to_layer_dict(self, layer_dict: dict, module: nn.Module,
                           recurse: Union[bool, int] = False) -> None:
        """Add the given module to the layer dictionary
        """
        for index, (name, child) in enumerate(module.named_children()):
            if not isinstance(module, nn.Sequential):
                layer_dict[name] = Layer(module=child, network=self, key=name)
            elif recurse:
                # FIXME[hack]
                print(f"torch Network: not adding layer: {index}, {name}, "
                      f"{type(child)}")

    def _layer_for_module(self, module: nn.Module) -> Layer:
        """Get the layer representing the given torch module.
        """
        for layer in self.layer_dict.values():
            if module is layer.module:
                return layer
        # pylint: disable=protected-access
        raise ValueError(f"Module {type(module)} ({module._get_name()}) "
                         "is not a Layer of this Network.")

    def _compute_layer_shapes(self, input_shape: tuple) -> None:
        """Compute the input and output shapes of all layers.
        The shapes are determined by propagating some dummy input through
        the network.

        This method will fill the private attributes _input_shapes
        and _output_shapes, mapping layer names to the respective
        shape.

        Arguments
        ---------
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

        # Send dummy data through the network to determine input and
        # output shapes of layers
        self._model(torch.zeros(*torch_input_shape, device=self._device))
        self._remove_hooks()

        self._prepare_layer_hooks(Layer.adapt_shapes)
        # Send dummy data through the network to determine input and
        # output shapes of layers
        self._model(torch.zeros(*torch_input_shape, device=self._device))
        self._remove_hooks()

    def _prepare_layer_hooks(self, hook,
                             layers: Iterable[Layer] = None) -> None:
        if layers is None:
            layers = self
        for layer in layers:
            module = layer.module
            bound_hook = getattr(layer, hook.__name__)
            self._hooks[module] = module.register_forward_hook(bound_hook)

    def _prepare_hooks(self, hook, modules=None) -> None:
        """Add hooks to the specified layers.
        """
        if modules is None:
            modules = self.layer_ids

        for module in modules:
            if isinstance(module, Layer):
                name = module.key
                module = module.module
            elif isinstance(module, str) and module in self.layer_dict:
                name = module
                module = self.layer_dict[module].module
            elif isinstance(module, nn.Module):
                name = str(module)  # FIXME[hack]
            else:
                raise ValueError(f"Illegal module specification: {module}")
            # pylint: disable=protected-access
            module._name = name  # FIXME[hack]
            self._hooks[module] = module.register_forward_hook(hook)

    def _remove_hooks(self, modules=None) -> None:
        if modules is None:
            modules = list(self._hooks.keys())

        for module in modules:
            if isinstance(module, Layer):
                module = module.module
            elif isinstance(module, str) and module in self.layer_dict:
                module = self.layer_dict[module].module
            elif not isinstance(module, nn.Module):
                raise ValueError(f"Illegal module specification: {module}")
            self._hooks[module].remove()
            del self._hooks[module]

    def _shape_hook(self, module: nn.Module, incoming, outgoing) -> None:
        """
        """
        name = module._name  # FIXME[hack]: how to acces the module name?
        # input[0].size() will be (N, C, H, W) -> store (H ,W, C)
        input_shape = incoming[0].size()
        self._input_shapes[name] = (None, *input_shape[2:], input_shape[1])
        # outgoing.size() will be (N, C, H, W) -> store (C, H ,W)
        output_shape = outgoing.size()
        self._output_shapes[name] = (None, *output_shape[2:], output_shape[1])

    def _activation_hook(self, module: nn.Module,
                         incoming, outgoing: Variable) -> None:
        # pylint: disable=unused-argument
        """
        """
        name = module._name  # FIXME[hack]: how to acces the module name?
        # outgoing is a torch.autograd.variable.Variable,
        # outgoing.data holds the actual data
        # FIXME[quesion]: it is said in the documentation, that
        # the TorchTensor and the numpy array share the same memory.
        # However, the TorchTensor may be removed after usage (is this true?)
        # what will then happen with the numpy array? do we need to
        # copy or is that a waste of resources?
        self._activations[name] = outgoing.data.cpu().numpy().copy()
        # if len(outgoing.data.size()) == 4:  # convolution, (N,C,H,W)
        #    self._activations[name] = \
        #        self._activations[name].transpose(0, 2, 3, 1)

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
        # pylint: disable=protected-access
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
        if self.layer_is_convolutional(first):
            return first.in_channels
        if (isinstance(first, nn.Sequential) and
                self.layer_is_convolutional(first[0])):
            return first[0].in_channels
        return 0  # no idea how to determine the number of input channels

    @property
    def layer_ids(self) -> list:
        """Get list of layer ids. These ids can be used to access layers via
        this Network API.

        Returns
        -------
        The list of layer identifiers.
        """
        # FIXME[hack]: we should avoid accessing protected members
        # pylint: disable=protected-access
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

    def get_layer_padding(self, layer_id) -> (int, int):
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

    def get_layer_output_padding(self, layer_id) -> (int, int):
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

    def get_layer_info(self, layername):
        """

        Parameters
        ----------
        layername

        Returns
        -------
        """
        # FIXME[todo]: implementation
        # (we still have to decide on some info API)

    @property
    def torch_device(self) -> torch.device:
        """The device on which this :py:class:`Network` (or more
        precisely the underlying torch model) is placed.
        """
        return self._device

    @property
    def model(self) -> nn.Module:
        """The underlying torch model.
        """
        return self._model

    @property
    def torch_model(self) -> nn.Module:
        """The underlying torch model.
        """
        return self._model

    @property
    def name(self) -> str:
        """The name of this model.
        """
        return self._model.__class__.__name__

    # -----------------------------------------------------------------------

    #
    # Applying the network to data
    #

    def _get_activations(self, input_samples: torch.Tensor,
                         layer_ids) -> List[torch.Tensor]:
        """Compute layer activations.
        """
        LOG.debug("TorchNetwork[%s]._get_activations: "
                  "inputs[%s]: %s (%s, %s), layers=%s", self,
                  type(input_samples).__name__,
                  getattr(input_samples, 'shape', '?'), self._internal_format,
                  getattr(input_samples, 'dtype', '?'), layer_ids)

        # prepare to record the activations
        self._activations = {}
        self._prepare_hooks(self._activation_hook, layer_ids)

        # run forward pass (this will fill self._activations)
        self._model(input_samples)

        # remove the hooks
        self._remove_hooks(layer_ids)

        for key, activations in self._activations.items():
            print(f"torch._get_activations[{key}]: {activations.shape}")

        # return the activations
        return list(self._activations.values())

    def _compute_net_input(self, layer_ids: list,
                           input_samples: np.ndarray) -> np.ndarray:
        """Computes a list of net inputs from a list of layer ids."""
        # FIXME[todo]: implementation

    def torch_forward(self, data: Datalike) -> np.ndarray:
        """Perform a forward step.
        """
        internal = self.data_to_internal(data)


    # FIXME[old]
    def _old_get_activations(self, input_samples: np.ndarray,
                             layer_ids) -> List[np.ndarray]:
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

        return super().get_activations(input_samples, layer_ids)

    #
    # Pre- and postprocessing
    #

    def _transform_input(self, inputs: np.ndarray,
                         data_format: str = None) -> Tuple[torch.Tensor,
                                                           bool, bool]:
        # FIXME[refactor]: we need to restructure this class and come
        # up with a better specialization/data conversion API that
        # correctly works with subclassing ...
        """Transform input data from the public interface (using numpy arrays)
        into the internal representation (`torch.Tensor`).

        Arguments
        ---------
        inputs: np.ndarray
            The input data.

        data_format: str
            The format of the input data. If `None`,
            the default format of the network (according to
            :py:prop:`data_format` is assumed).

        Result
        ------
        internal: torch.Tensor
            The data in the internal format

        batched: bool
            A flag indicating that a batch dimension was added.

        internalized: bool
            A flag indicating if the data was actively transformed
            into the internal format (torch.Tensor). This is `True`,
            if an active transformation was performed and `False`
            if the given `inputs` were already a `torch.Tensor`.

        """

        if isinstance(inputs, torch.Tensor):
            # The data seems already to be in interal format.
            internalized = False
            batched = False

        else:
            inputs, batched, internalized = \
                super()._transform_input(inputs, data_format=data_format)
            internalized = True

            inputs = torch.from_numpy(inputs)
            inputs = inputs.to(self._device)
            inputs = Variable(inputs)

        return inputs, batched, internalized

    # FIXME[old]:
    def _old_postprocess(self, layer_ids, input_samples):
        # if no batch data was provided, remove batch dimension from
        # activations
        if (input_samples.shape[0] == 1
                and len(input_samples.shape) < 4
                and input_samples.shape[0] != 1):
            for key in self._activations:
                self._activations[key] = self._activations[key].squeeze(0)

        # if a single layer_id was given (not a list), then just return
        # a single activtion array
        return ([self._activations[_] for _ in layer_ids]
                if isinstance(layer_ids, list)
                else self._activations[layer_ids])

    def numpy_to_internal(self, array: np.ndarray) -> torch.Tensor:
        # FIXME[refactor]: we need to restructure this class and come
        # up with a better specialization/data conversion API that
        # correctly works with subclassing ...
        """Translate a numpy array into the internal representation,
        that is a torch tensor.
        """
        tensor = torch.from_numpy(array)
        tensor = tensor.to(self._device)
        return tensor

    def internal_to_numpy(self, tensor: torch.Tensor,
                          unbatch: bool = False) -> np.ndarray:
        # FIXME[refactor]: we need to restructure this class and come
        # up with a better specialization/data conversion API that
        # correctly works with subclassing ...
        # pylint: disable=arguments-renamed
        """Translate the internal representation (torch tensor) into
        a numpy array.
        """
        if unbatch:
            if tensor.shape[0] != 1:
                raise ValueError("Cannot unbatch scores "
                                 f"with batch size {tensor.shape[0]}")
            tensor = tensor[0]
        tensor = tensor.to('cpu')
        return tensor.numpy()


class ImageNetwork(BaseImageNetwork, Network):
    """A torch :py:class:`Network` operating on images.
    """

    internal_result: Tuple[str, ...] = ('result', )
    external_result: Tuple[str, ...] = ('result', )

    # the torch ImageNet dataset mean and std will remain the same
    # irresptive of the torchvision model used. These values assume
    # that ImageNet images are RGB and have range 0.0 to 1.0.
    imagenet_mean_ = torch.FloatTensor([0.485, 0.456, 0.406])
    imagenet_std_ = torch.FloatTensor([0.229, 0.224, 0.225])

    # the lower and upper bound for image values, assuming that images
    # are normalized to zero mean and unit variance using the ImageNet
    # mean and standard deviation.
    imagenet_normalized_min_ = -imagenet_mean_/imagenet_std_
    imagenet_normalized_max_ = (1-imagenet_mean_)/imagenet_std_

    lower_bound = imagenet_normalized_min_
    upper_bound = imagenet_normalized_max_

    # Construct an image preprocessing function using
    # torch.transforms.  Notice that (at least transforms.Resize)
    # expects as input a PIL image (not a numpy.ndarray).
    # The result will be of type torch.Tensor.
    _preprocess_pil_image = transforms.Compose([
        transforms.Resize(256),  # requires PIL.Image
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean_, std=imagenet_std_)
    ])

    preprocess_pil = transforms.Compose([
        transforms.Resize((299, 299)),  # requires PIL.Image
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean_, std=imagenet_std_)
    ])

    preprocess_numpy = transforms.Compose([
        # transforms.Resize((299, 299)),  # requires PIL.Image
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean_, std=imagenet_std_)
    ])

    def __init__(self, model: Optional[Modellike] = None, **kwargs):
        super().__init__(model=model, **kwargs)

    def _image_to_internal(self, image: PIL.Image) -> torch.Tensor:
        """
        """
        return self._preprocess_pil_image(image)

    # FIXME[hack]: image_to_internal of super class should made more general,
    # so that overwriting that method is not necessary.
    # This ad hoc function only works for a single image, not for a batch
    # of images!
    def image_to_internal(self, image: Imagelike) -> torch.Tensor:
        """Transform an image into a torch Tensor.
        """
        if isinstance(image, torch.Tensor):
            image_tensor = image

        elif isinstance(image, np.ndarray):
            # at this point we need to know the range (if further
            # preprocessing, e.g., normalization, is required ...)
            if False and (0 <= image).all():
                if (image <= 1).all():  # image is range 0.0 - 1.0
                    pass
                elif (image <= 255).all():   # image is range 0 - 255
                    pass

            # Question: channel first or channel last?
            # H X W X C  ==>   C X H X W
            #  image = np.transpose(image, (2, 0, 1))

            # preprocess_numpy expects numpy.ndarray of correct size,
            # dtype float and values in range [0.0, 1.0].
            # It performs the following operations:
            #  1. [no resizing]
            #  2. numpy.ndarray -> torch.Tensor
            #  3. normalization [0.0, 1.0] -> torch.imagenet_range
            image_tensor = self.preprocess_numpy(image)

            # old: explicit transformation:
            # H X W X C  ==>   C X H X W
            # image = np.transpose(image, (2, 0, 1))
            #
            # image = torch.from_numpy(image)
            # image = image.add(-self.imagenet_mean_.view(3, 1, 1)).\
            #     div(self.imagenet_std_.view(3, 1, 1))
            #
            # add batch dimension:  C X H X W ==> B X C X H X W
            # image = image.unsqueeze(0)

        else:
            # the _image_to_internal function expects as input a PIL image!
            image = Image.as_pil(image)

            # image should be PIL Image. Got <class 'numpy.ndarray'>
            image_tensor = self._image_to_internal(image)

        if image_tensor.dim() == 4:
            # image is already batch
            image_batch = image_tensor
        elif image_tensor.dim() == 3:
            # create a mini-batch as expected by the model
            # by adding a batch dimension:  C X H X W ==> B X C X H X W
            image_batch = image_tensor.unsqueeze(0)
        else:
            raise ValueError(f"Data of invalid shape {image.shape} cannot "
                             "be transformed into an internal torch image.")

        # move the input and model to GPU for speed if available
        image_batch = image_batch.to(self._device)

        return image_batch

    def internal_to_image(self, data: torch.Tensor,
                          dtype: type = float) -> Imagelike:
        """Transform a data tensor into a numpy array, undoing the
        :py:meth:`image_to_internal` operations.  This includes:

        1. removing the batch axis (if batch size is 1)
        2. unnormalizing: torch imagenet_range -> [0.0, 1.0]
        3. convert to numpy
        4. reorder: channel first to channel last
        """
        
        # remove batch dimension # B X C H X W ==> C X H X W
        data = data.squeeze(0)

        # "unnormalize": reverse of normalization operation
        data = data.cpu().\
            mul(self.imagenet_std_.view(3, 1, 1)).\
            add(self.imagenet_mean_.view(3, 1, 1))
        # clip values to the valid range
        data = torch.clip(data, 0, 1)
        # numpy equivalent: data = np.clip(data, 0, 1)

        # C X H X W  ==>   H X W X C
        data = data.permute(1, 2, 0)
        # numpy equivalent: data = np.transpose(data, (1,2,0))

        # to numpy
        data = data.numpy()
        if issubclass(dtype, int):
            data = (data*255).astype(np.uint8)

        return data

    def range_info(self):
        """Output the range of values expected as inputs to this model.
        """
        range_min = self.lower_bound.numpy()
        range_max = self.upper_bound.numpy()
        delta = range_max-range_min
        print("Images are transformed to the following range:\n"
              f"  red:   {range_min[0]:.4f} - {range_max[0]:.4f}"
              f"  (range={delta[0]:.4f}, quantization={delta[0]/256:.4f})\n"
              f"  green: {range_min[1]:.4f} - {range_max[1]:.4f}"
              f"  (range={delta[1]:.4f}, quantization={delta[1]/256:.4f})\n"
              f"  blue:  {range_min[2]:.4f} - {range_max[2]:.4f}"
              f"  (range={delta[2]:.4f}, quantization={delta[2]/256:.4f})")

    def _process(self, images: torch.Tensor) -> torch.Tensor:
        # FIXME[refactor]: we need to come up with a better processing scheme
        # to subclass the Tool class in a type save way.
        # pylint: disable=arguments-differ
        return self._model(images)


class ClassifierNetwork(BaseClassifier, Network):
    """A torch realization of a classifier network.
    """

    #
    # Implementation of the SoftClassifier API
    #

    # FIXME[todo]: it should be possible to use the functionalitly
    # implemented in the parent class (network.Classifier), if
    # we could provide the logit_layer and score_layer properties

    def class_scores(self, inputs: Datalike) -> np.ndarray:
        """Get the class confidence scores for the given image.
        """
        # FIXME[hack]: this does not allow to be called with batches!
        scores = self._class_scores(self.image_to_internal(inputs))
        return scores[0].cpu().detach().numpy()

    def _class_scores(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self._model(inputs)
        # Tensor of shape 1000, with confidence scores over Imagenet's
        # 1000 classes
        # print(output[0])

        # The output has unnormalized scores. To get probabilities,
        # we run a softmax on it.
        scores = torch.nn.functional.softmax(logits, dim=1)
        return scores

    def _old_classify(self, inputs: Datalike, top=False,
                      output=None) -> ClassIdentifier:
        """Output the top-n classes for given batch of inputs.

        Arguments
        ---------
        inputs: np.ndarray
            A data point or a batch of input data to classify.

        Results
        -------
        classes:
            A list of class-identifiers or a
            list of tuples of class identifiers.
        """
        return self._predict(inputs, top=top, output=output)

    # FIXME[todo]: this method is designed for classification and should
    # probably be better integrated with the other methods of this
    # class and the super classes
    def _predict(self, data: torch.Tensor, top: Union[int, bool] = False,
                 output=None) -> Tuple[np.ndarray, np.ndarray]:
        """Do the actual prediction.

        Arguments
        ---------
        output:
            How to report the confidence values: None means that
            no confidence values are reported, 'logits' means that
            logit value are returned, while 'probs' will return
            probability values (obtained by applying softmax).
        """
        logits = self._model.forward(data.to(self._device))

        if not top:
            indices = int(torch.argmax(logits, dim=1)[0])
        else:
            indices = torch.argsort(logits, dim=1,
                                    descending=True)[:, :top].numpy()

        if output is None:
            return indices
        if output == 'logits':
            return indices, logits[:, indices].numpy()
        if output == 'probs':
            probs = F.softmax(logits, dim=1)
            # confidence = round(float(torch.max(probs.data, 1)[0][0]) * 100, 4)
            return indices, probs.data[0, indices].cpu().numpy()

        raise ValueError(f"Invalid value for argument: output='{output}'")

    def _old_classify_image(self, image: Imagelike,
                            preprocess: bool = True, top: int = 5) -> None:
        """Classify an image an output result(s).
        """

        if preprocess:
            data = self.old_preprocess_image(image)
        else:
            data = image

        # model predictions
        indices, confidences = \
            self._predict(data, top=top, output='probs')

        # get an index(class number) of a largest element
        # confidence, label_idx = torch.max(output.data, dim=1)
        for i, (index, confidence) in \
                enumerate(zip(indices[0], confidences[0])):
            label = self.get_label(index, name='text')
            print(f"  {i+1}). Label: {label} ({index}), "
                  f"confidence={confidence*100:2.2f}%")
