from network import BaseNetwork

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from collections import OrderedDict
from typing import List
import importlib.util

## FIXME[todo]: not finished yet - will not run!
## FIXME[todo]: integrate with base class
## FIXME[todo]: check docstrings
## FIXME[todo]: not tested yet!
## FIXME[todo]: need to clean up

## FIXME[todo]: cuda activation (if available)
## FIXME[todo]: channel last (like in RGB images)


## * tensorflow default is to use channel last ("NHWC")
##   (can be changed to channel first: data_format="NCHW")
##   https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
##
## * pytorch only supports channel first (N,C,H,W)
##   http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
##
## * ruediger claims that cuDNN requires channel first
##
## * default for RGB images seems to be channel last (H,W,C)


class TorchNetwork(BaseNetwork):
    """
    A class implmeenting the network interface (BaseNetwork)
    to access feed forward networks implemented in (py)Torch.

    Some general remarks on (py)torch:

    * Torch convolution follows the channel first scheme, that is
      the shape of a 2D convolution is (batch, channel, height, width).

    * Torch does not store activation values. However, one can
      register hooks to be executed before or after the forward or
      backward propagation.  These hooks can be use to access and
      store input and output values.
    """

    _model : nn.Module
    _input_shapes : dict = None
    _output_shapes : dict = None
    _hooks : dict = {}

    def __init__(self, *args, **kwargs):
        """
        Load Torch model.
        
        Parameters
        ----------
        model_file
            Path to the .h5 model file.
        """

        i = 0
        data_loaded = False

        ##
        ## Try to get a model:
        ##

        if len(args) < 1:
            raise TypeError("TorchNetwork requires at least one argument")
        
        if isinstance(args[i], nn.Module):
            self._model = args[i]
            i += 1
        elif isinstance(args[i], str) and args[i].endswith(".pth.tar"):
            self._model = torch.load(args[i])
            data_loaded = True
            i += 1
        elif isinstance(args[i], str) and args[i].endswith(".py"):
            spec = importlib.util.spec_from_file_location('torchnet', args[i])
            net_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(net_module)
            net_class_name = kwargs.get('network_class', 'Net')
            net_class = getattr(net_module, net_class_name) 
            self._model = net_class()
            i += 1
        else:
            raise ValueError("Invalid arguments for constructing a TorchNetwork")

        ##
        ## Load model parameter if specified:
        ##
        
        if i < len(args) and not data_loaded:
            if isinstance(args[i], str) and args[i].endswith(".pth"):
                self._model.load_state_dict(torch.load(args[i]))

        ## FIXME[todo]: set training/test state
        #self._model.train()
        #self._model.eval()

        ## FIXME[todo]: activate gpu support if available
        # self._use_cuda = torch.cuda.is_available()
        # if self._use_cuda:
        #    self._model.cuda()

        if 'input_shape' in kwargs:
            self._compute_layer_shapes(kwargs['input_shape'])


    def _compute_layer_shapes(self, input_shape : tuple) -> None:
        """Compute the input and output shapes of all layers.
        The shapes are determined by probagating some dummy input through
        the network."""
        
        first_layer = self._get_first_layer()
        
        if self.is_convolution(first_layer):
            if input_shape[-1] != first_layer.in_channels:
                input_shape = (*input_shape,first_layer.in_channels)
            if len(input_shape) < 4:
                input_shape = (1,*input_shape)
            input_shape = (input_shape[0],input_shape[-1],*input_shape[1:-1])
        elif len(input_shape) < 2:
            input_shape = (1,*input_shape)
        if input_shape[0] != 1:
            input_shape = (1,*input_shape[1:])

        
        print("_compute_layer_shapes: input_shape={}".format(input_shape))
        self._input_shapes = {}
        self._output_shapes = {}
        self._prepare_hooks(self._shape_hook)
        _ = self._model(Variable(torch.zeros(*input_shape), volatile=True))
        self._remove_hooks()

    def _compute_layer_shapes_old(self, input_shape : tuple) -> None:
        """Compute the input and output shapes of all layers,
        starting from an input signal of the given shape.
        """

        raise NotImplementedError("old code, not for use anymore")
        ## Remark: probably it is more reliable to determine the
        ## shapes once some real data is propagated through the
        ## network (using some hook). The drawback is, that one can
        ## not access this information before some 
        
        ## Remark: In general it is not possible to compute the input
        ## and output shape for convolutional layers just from the
        ## following dense layers. There are different theoretical
        ## possibilities (unless one assumes a quadratic shape, which
        ## we should not do as it reduces generality).
        
        ## According to the torch documentation, the output shape of a
        ## 2D convolutional layer can be computed from the input shape
        ## by the following formulae:
        ##
        ##  h_out = floor((h_in
        ##                 + 2*padding[0]
        ##                 - dilation[0]*(kernel_size[0]-1)
        ##                 -1) / stride[0] + 1)
        ##  w_out = floor((w_in
        ##                 + 2*padding[1]
        ##                 - dilation[1]*(kernel_size[1]-1)
        ##                 -1) / stride[1] + 1)
        ##
        ## [http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d]

        ## FIXME[bug]: this function will give wrong values if max
        ## pooling or some other kind of shape changing operation is
        ## performed in the forward() method.

        ## FIXME[todo]: Possible solution: just create some dummy
        ## input of apropriate input shape and propagate it through
        ## the network to record the given shapes.
        
        #for name, module in self._model._modules.items():
        #    if self.is_convolution(name):
        #        self._input_shapes[name] = input_shape
        #        out_channels = module.out_channels
        #        out_h = floor((input_shape[0]
        #                       + 2 * module.padding[0]
        #                       - module.dilation[0]
        #                       * (module.kernel_size[0]-1) -1)
        #                      / moduel.stride[0] + 1)
        #        out_w = floor((input_shape[1]
        #                       + 2 * module.padding[1]
        #                       - module.dilation[1]
        #                       * (module.kernel_size[1]-1) -1)
        #                      / moduel.stride[1] + 1)
        #        self._output_shapes[name] = (out_h, out_w, out_channels)
        #    else:
        #        self._input_shapes[name] = (module.in_features,)
        #        self._output_shapes[name] = (module.out_features,)
        #    input_shape = self._output_shapes[name]

    def _prepare_hooks(self, hook, layer_ids = None) -> None:
        print('TorchNetwork._prepare_hooks: info: called with layer_ids = {}'.format(layer_ids))

        if layer_ids is None:
            layer_ids = self.layer_ids
        print('TorchNetwork._prepare_hooks: info: using layer_ids = {}'.format(layer_ids))
    
        for id in layer_ids:
            print('TorchNetwork._prepare_hooks: info: going to add hook for module "{}"'.format(id))
            module = self._model._modules[id]
            module._name = id # FIXME[hack]: how to acces the module name?
            self._hooks[id] = module.register_forward_hook(hook)
            
    def _remove_hooks(self, layer_ids = None) -> None:
        if layer_ids is None:
            layer_ids = list(self._hooks.keys())
        for id in layer_ids:
            self._hooks[id].remove()
            del self._hooks[id]
            print('TorchNetwork._remove_hooks: info: removed hook from module "{}"'.format(id))


    def _shape_hook(self, module, input, output):
        name = module._name # FIXME[hack]: how to acces the module name?
        # input[0].size() will be (N, C, H, W) -> store (H ,W)
        self._input_shapes[name] = tuple(input[0].size()[1:])
        # output.size() will be (N, C, H, W) -> store (C, H ,W)
        self._output_shapes[name] = tuple(output.size()[1:])
        
    def _activation_hook(self, module, input, output):
        name = module._name # FIXME[hack]: how to acces the module name?
        # output is a torch.autograd.variable.Variable,
        # output.data holds the actual data
        # FIXME[quesion]: it is said in the documentation, that
        # the TorchTensor and the numpy array share the same memory.
        # However, the TorchTensor may be removed after usage (is this true?)
        # what will then happen with the numpy array? do we need to
        # copy or is that a waste of resources?
        self._activations[name] = output.data.numpy().copy()
        if len(output.data.size()) == 4: # convolution, (N,C,H,W)
            self._activations[name] = self._activations[name].transpose(0,2,3,1)

    def _general_hook(self, module, input, output):
        self._shape_hook(module, input, output)
        self._activation_hook(module, input, output)
        

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
        return self._model._modules[layer_id]

    def _get_first_layer(self) -> nn.Module:
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


    def is_convolution(self, layer) -> bool:
        """Check if the given layer is a convolutional layer. If so,
        additional information can be obtained by the methods
        get_kernel_size, get_channels, get_stride, get_padding, and
        get_dilation.

        Parameters
        ----------
        layer:
            Identifier of a layer in this network.

        Returns
        -------
        The layer for the given identifier.
        """
        if not isinstance(layer,torch.nn.Module):
            layer = self._get_layer(layer)
        return isinstance(layer,nn.modules.conv.Conv2d)


    def get_units(self, layer_id) -> int:
        """The number of output units of this layer.
        
        Parameters
        ----------
        layer_id:
            Identifier of a layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a layer in this network.
        """
        
        
    def get_kernel_size(self, layer_id) -> int:
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
        if self.is_convolution(layer_id):
            layer = self._get_layer()
            return layer.kernel_size
        else:
            raise ValueError('No kernel size for "{}"'.format(layer_id))


    def get_input_channels(self, layer_id) -> int:
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
        if self.is_convolution(layer_id):
            layer = self._get_layer()
            return layer.in_channels
        else:
            raise ValueError('No input channels for "{}"'.format(layer_id))


    def get_output_channels(self, layer_id) -> int:
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
        if self.is_convolution(layer_id):
            layer = self._get_layer()
            return layer.out_channels
        else:
            raise ValueError('No output channels for "{}"'.format(layer_id))


    def get_stride(self, layer_id) -> (int, int):
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
        if self.is_convolution(layer_id):
            layer = self._get_layer()
            return layer.stride
        else:
            raise ValueError('No stride for "{}"'.format(layer_id))


    def get_padding(self, layer_id) -> (int,int):
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
        if self.is_convolution(layer_id):
            layer = self._get_layer()
            return layer.padding
        else:
            raise ValueError('No padding for "{}"'.format(layer_id))


    def get_output_padding(self, layer_id) -> (int,int):
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
        if self.is_convolution(layer_id):
            layer = self._get_layer()
            return layer.output_padding
        else:
            raise ValueError('No output padding for "{}"'.format(layer_id))


    def get_dilation(self, layer_id) -> (int, int):
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
        if self.is_convolution(layer_id):
            layer = self._get_layer()
            return layer.dilation
        else:
            raise ValueError('No dilation for "{}"'.format(layer_id))

        
        
    # FIXME[design]: we should decide for a standard way to encode
    # (e.g. convolution as (channel, width, height) or (width, height, channel), or (batch, height, channel), ...
    def get_layer_input_shape(self, layer_id) -> tuple:
        """
        Give the shape of the input of the given layer.

        Parameters
        ----------
        layer_id

        Returns
        -------
        (units) for dense layers
        (width, height, channels) for convolutional layers

        Raises
        ------
        Runtim
        """
        return self._get_layer(layer_id).get_input_shape_at(0)


    # FIXME[design]: we should decide for a standard way to encode
    def get_layer_output_shape(self, layer_id) -> tuple:
        """
        Give the shape of the output of the given layer.

        Parameters
        ----------
        layer_id

        Returns
        -------

        """
        return self._get_layer(layer_id).get_output_shape_at(0)


    # FIXME[design]: why list and not pair (weights, bias)?
    # or two functions: weights and bias?
    def get_layer_weights(self, layer_id) -> List[np.ndarray]:
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
            Weights of the layer.

        """
        return self._get_layer(layer_id).get_weights()



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

        """


        if self._input_shapes is None:
            self._compute_layer_shapes(input_samples.shape)
        input_channels = 1 # FIXME[todo]: get from first layer

        if (input_channels == 1 and
            len(input_samples.shape) < 4 and
            input_samples.shape[-1] != 1):
            input_samples = np.expand_dims(input_samples, axis=-1)
        no_batch = (len(input_samples.shape) == 3)
        if no_batch:
            input_samples = np.expand_dims(input_samples, axis=0)

        ## sanity check
        if (len(input_samples.shape) != 4 or
            input_samples.shape[-1] != input_channels):
            raise ValueError("Input data have inapropriate shape: should be (batch, height, width, channel).")

        ## pytorch expects  channel first (N,C,H,W)
        input_samples = input_samples.transpose((0,3,1,2)).copy()
        torch_input =  Variable(torch.from_numpy(input_samples), volatile=True)

        ## use GPU
        # if self._use_cuda:
        #    torch_input = torch_input.cuda()

        ## prepare to record the activations
        self._activations = {}
        self._prepare_hooks(self._activation_hook, layer_ids)

        torch_output = self._model(torch_input)

        self._remove_hooks(layer_ids)

        if no_batch:
            for id in self._activations.keys():
                self._activations[id] = self._activations[id].squeeze(0)
                                       
        return [self._activations[_] for _ in layer_ids]


        # FIXME[todo]: there does not seem to be a standard way to
        #
        # Torch problems:
        #
        # * how to get layer_id?
        #   -got myModuleIndex from visual (using the __tostring() metamethod).
        # model:get(myModuleIndex).output

        # you can access to a module with net.modules[n] (n is the
        # index of the module, use print(net) to see your whole
        # network and its modules). Then each module has to state
        # variables output and gradInput (gradient of the module with
        # respect to its input), then you can access the output of the
        # nth intermediate layer with
        #    net.modules[n].output

        # Every layer in torch's models can be accessed using model.modules.
        # Once you have made a forward pass, like this :
        #   input  = torch.Tensor(5, 10)
        #   model:forward(input)
        # You can view the output dimension of each layer.
        # If you want to view for a particular layer,
        # say the second layer in the above case, you can simply do:
        #   model.modules[2].output:size()
        # _modules.items()
        # _modules.keys()



from torchvision import datasets, transforms
import torch.nn.functional as F
import os

def main():
    #model_file = "models/example_torch_mnist_model.pth.tar"
    #network = TorchNetwork(model_file)
    net_file = "models/example_torch_mnist_net.py"
    parameter_file = "models/example_torch_mnist_model.pth"
    network = TorchNetwork(net_file, parameter_file, net_class = 'Net', input_shape = (28,28))
    print("main: {}".format(network.layer_ids))
    print("main: input_shapes(1) = {}".format(network._input_shapes))
    print("main: output_shapes(1) = {}".format(network._output_shapes))

    batch_size = 1000
    datadir = os.getenv('MNIST_DATA', '../data')
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(datadir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=batch_size, shuffle=True)

    data, target = next(test_loader.__iter__())
    
    
    input_data = data.numpy()
    print(input_data.shape)
    input_data = input_data[0].squeeze(0)
    print(input_data.shape)
    activations = network.get_activations(("conv1", "conv2"), input_data)
    print("main: type={}, len={}".format(type(activations), len(activations)))
    print("main: result 0: {} {}".format(type(activations[0]), activations[0].shape))
    print("main: result 1: {} {}".format(type(activations[1]), activations[1].shape))
    
if __name__ == "__main__": main()
