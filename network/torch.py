
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
from typing import Tuple, Union, List, Iterable
from collections import OrderedDict
import logging

import importlib.util

# third party imports
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

# toolbox imports
from dltb.base.data import Data
from dltb.base.image import Image, Imagelike, ImageAdapter
from dltb.network.network import Network as BaseNetwork
from dltb.network.network import ImageNetwork as BaseImageNetwork
from dltb.network import layer as Base
from dltb.util.array import DATA_FORMAT_CHANNELS_FIRST
from . import Layer as BaseLayer, Classifier


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
        LOG.info(f"torch: new Layer with type %s", type(module))
        if isinstance(module, nn.Conv2d):
            cls = Conv2D
        elif isinstance(module, nn.Sequential):
            cls = Sequential

        if super().__new__ is object.__new__:
            # object.__new__() takes exactly one argument
            return object.__new__(cls)
        return super().__new__(cls, **kwargs)

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

        print(f"Created new torch layer: {self} ({type(module)}): "
              f"{self.input_shape}, {self.output_shape}")

    def adapt_shapes(self, module: nn.Module,
                     input: torch.Tensor, output: torch.Tensor) -> None:
        input_shape = input[0].size()
        output_shape = output.size()
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
        return self._weight.numpy()

    @property
    def bias(self) -> np.ndarray:
        return self._bias.numpy()


class StridingLayer(Layer, Base.StridingLayer):

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
        return self._ops[0].outputs[0]


class Dense(NeuralLayer, Base.Dense):
    pass


class Conv2D(NeuralLayer, StridingLayer, Base.Conv2D):

    @property
    def kernel_size(self):
        return self._module.kernel_size

    @property
    def dilation(self):
        print(f"torch.Conv2D: dilation={self._module.dilation}")
        return self._module.dilation

    @property
    def filters(self):
        return self.output_shape[1]

    @property
    def weight_tensor(self):
        # The last input of the first operation with strides
        # should correspond to the (first part of the) weights.
        striding_op = self._first_module_with_attr('strides')
        return striding_op.inputs[-1]


class MaxPooling2D(StridingLayer, Base.MaxPooling2D):

    @property
    def pool_size(self):
        return self._module.kernel_size

    @property
    def dilation(self):
        return self._module.dilation


class Dropout(Layer, Base.Dropout):
    pass


class Flatten(Layer, Base.Flatten):
    pass


# -----------------------------------------------------------------------------


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

    # pytorch expects channel first (batch,channel,height,width)
    _internal_format = DATA_FORMAT_CHANNELS_FIRST

    # _model: the underlying torch.nn.Module
    _model: nn.Module

    # _hooks: a dictionary mapping layer names to hooks
    _hooks: dict = {}

    # _python_class: a sublcass of :py:class:`torch.nn.Module` that
    #   will be used to initialize a torch.Network upon preparation.
    _python_class: type = None

    # _torch_filename: name of the weights file (either .pth or .pth.tar)
    _torch_filename: str = None

    # _device: the device to perform computations on
    #   ('cpu', 'cuda', or 'cuda0')
    _device: torch.device = None

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
        if cuda:
            self._device = torch.device('cuda')  # 'cuda:0'
        else:
            self._device = torch.device('cpu')

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

        self._model.to(self._device)

        # set train/eval mode
        if self._train_mode:
            self._model.train()
        else:
            self._model.eval()

        super()._prepare()

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
            # print(f"adding layer: ", index, name, type(child))
            layer_dict[name] = Layer(module=child, network=self, key=name)

    def _layer_for_module(self, module: nn.Module) -> Layer:
        """Get the layer representing the given torch module.
        """
        for layer in self._layer_dict.values():
            if module is layer._module:
                return layer
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
            module = layer._module
            bound_hook = getattr(layer, hook.__name__)
            self._hooks[module] = module.register_forward_hook(bound_hook)

    def _prepare_hooks(self, hook, modules=None) -> None:
        """Add hooks to the specified layers.
        """
        if modules is None:
            modules = self.layer_ids

        for module in modules:
            if isinstance(module, Layer):
                name = module.get_id()
                module = module._module
            elif isinstance(module, str) and module in self.layer_dict:
                name = module
                module = self.layer_dict[module]._module
            elif isinstance(module, nn.Module):
                name = str(module)  # FIXME[hack]
            else:
                raise ValueError(f"Illegal module specification: {module}")
            module._name = name  # FIXME[hack]
            self._hooks[module] = module.register_forward_hook(hook)

    def _remove_hooks(self, modules=None) -> None:
        if modules is None:
            modules = list(self._hooks.keys())

        for module in modules:
            if isinstance(module, Layer):
                module = module._module
            elif isinstance(module, str) and module in self.layer_dict:
                module = self.layer_dict[module]._module
            elif not isinstance(module, nn.Module):
                raise ValueError(f"Illegal module specification: {module}")
            self._hooks[module].remove()
            del self._hooks[module]

    def _shape_hook(self, module, input, output):
        name = module._name  # FIXME[hack]: how to acces the module name?
        # input[0].size() will be (N, C, H, W) -> store (H ,W, C)
        input_shape = input[0].size()
        self._input_shapes[name] = (None, *input_shape[2:], input_shape[1])
        # output.size() will be (N, C, H, W) -> store (C, H ,W)
        output_shape = output.size()
        self._output_shapes[name] = (None, *output_shape[2:], output_shape[1])

    def _activation_hook(self, module, input, output):
        name = module._name  # FIXME[hack]: how to acces the module name?
        # output is a torch.autograd.variable.Variable,
        # output.data holds the actual data
        # FIXME[quesion]: it is said in the documentation, that
        # the TorchTensor and the numpy array share the same memory.
        # However, the TorchTensor may be removed after usage (is this true?)
        # what will then happen with the numpy array? do we need to
        # copy or is that a waste of resources?
        self._activations[name] = output.data.cpu().numpy().copy()
        #if len(output.data.size()) == 4:  # convolution, (N,C,H,W)
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
        return first.in_channels if self.layer_is_convolutional(first) else 0

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

    def _get_activations(self, inputs: torch.Tensor,
                         layer_ids) -> List[torch.Tensor]:
        """Compute layer activations.
        """
        LOG.debug("TorchNetwork[%s]._get_activations: "
                  "inputs[%s]: %s (%s, %s), layers=%s", self.key,
                  type(inputs).__name__,
                  getattr(inputs, 'shape', '?'), self._internal_format,
                  getattr(inputs, 'dtype', '?'), layer_ids)

        # prepare to record the activations
        self._activations = {}
        self._prepare_hooks(self._activation_hook, layer_ids)

        # run forward pass (this will fill self._activations)
        self._model(inputs)

        # remove the hooks
        self._remove_hooks(layer_ids)

        for key, activations in self._activations.items():
            print(f"torch._get_activations[{key}]: {activations.shape}")

        # return the activations
        return list(self._activations.values())

    def _transform_input(self, inputs: np.ndarray,
                         **kwargs) -> Tuple[torch.Tensor, bool, bool]:
        """
        Transform input data from the public interface (using numpy arrays)
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
                super()._transform_input(inputs, **kwargs)
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
            for id in self._activations.keys():
                self._activations[id] = self._activations[id].squeeze(0)

        # if a single layer_id was given (not a list), then just return
        # a single activtion array
        return ([self._activations[_] for _ in layer_ids]
                if isinstance(layer_ids, list)
                else self._activations[layer_ids])

    def numpy_to_internal(self, array: np.ndarray) -> torch.Tensor:
        """
        """
        tensor = torch.from_numpy(array)
        tensor = tensor.to(self._device)
        return tensor

    def internal_to_numpy(self, tensor: torch.Tensor,
                           unbatch: bool = False) -> np.ndarray:
        """
        """
        if unbatch:
            if tensor.shape[0] != 1:
                raise ValueError("Cannot unbatch scores "
                                 f"with batch size {tensor.shape[0]}")
            tensor = tensor[0]
        tensor = tensor.to('cpu')
        return tensor.numpy()


class ImageNetwork(BaseImageNetwork, Network):

    def _prepare(self) -> None:
        super()._prepare()

        # Construct an image preprocessing function using torch.transforms.
        # Notice that (at least resize) expects as input a PIL image
        # (not a numpy.ndarray).
        self._preprocess_image = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def image_to_internal(self, image: Imagelike) -> torch.Tensor:
        """Transform an image into a torch Tensor.
        """
        # the _preprocess_image function expects as input a PIL image!
        image = Image.as_pil(image)

        # image should be PIL Image. Got <class 'numpy.ndarray'>
        image_tensor = self._preprocess_image(image)

        # create a mini-batch as expected by the model
        image_batch = image_tensor.unsqueeze(0)

        # move the input and model to GPU for speed if available
        image_batch = image_batch.to(self._device)

        return image_batch

    def internal_to_image(self, data: torch.Tensor) -> Imagelike:
        """Transform a torch Tensor into an image.
        """
        return data


class DemoResnetNetwork(Classifier, ImageNetwork):
    """An experimental Torch Network based on ResNet.

    https://pytorch.org/hub/pytorch_vision_resnet/
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(scheme='ImageNet', lookup='torch',
                         input_shape=(3, 224, 244),  # FIXME[hack]: can be different
                         **kwargs)

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
        super()._prepare()


    #
    # Implementation of the SoftClassifier interface
    #

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
