"""Access to the `lucid` activation maximization package.

`lucid.modelzoo.vision_models`:
    A module providing the pretrained networks by name, e.g.
    vision_models.AlexNet

"""
# standard imports
from typing import Iterator
from collections import OrderedDict
import logging

# thirdparty imports
from lucid.modelzoo import nets_factory
from lucid.modelzoo.vision_base import Model as LucidModel

# toolbox imports
from .tensorflow.network import Network as TensorflowNetwork

# logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


class Network(TensorflowNetwork):
    """A LucidModel seems to have a similar purpose as our
    :py:class:`Network` class. Hence we will subclass the
    :py:class:`tensorflow.Network` to get a LucidNetwork.

    The :py:class:`LucidModel` provides the following methods:

    :py:meth:`LucidModel.create_input` creates an input tensor,
    actually a pair

        (<tf.Tensor 'Placeholder:0' shape=(227, 227, 3) dtype=float32>,
          <tf.Tensor 'add:0' shape=(1, ?, ?, 3) dtype=float32>)

    The method can be called multiple times and will each time
    create a new Placeholder.

    _model: LucidModel
    _name: str

    _dataset: str
        'ImageNet'

    """

    def __init__(self, model: LucidModel = None, name: str = None, **kwargs):
        self._model = \
            nets_factory.models_map[name]() if model is None else model
        self._name = name
        LOG.debug(f"New LucidNetwork({name}) {self._model}")
        # load the graph definition (tf.GraphDef) from a binary
        # protobuf file and reset all devices in that GraphDef.
        self._model.load_graphdef()

        super().__init__(graph_def=self._model.graph_def, **kwargs)

    @property
    def model(self) -> LucidModel:
        """The underlying lucid model.
        """
        return self._model

    @property
    def name(self) -> str:
        """The name of the lucid model.
        """
        return self._name

    def _create_layer_dict(self) -> OrderedDict:
        """Try to find the sequences in operations of the graph that
        match the idea of a layer.

        Returns
        -------
        A mapping of layer_ids to layer objects.
        """
        layer_dict = OrderedDict()

        # FIXME[hack]: create a layer dict
        # Here we can use the model.layers dictionary from the Lucid model.
        LOG.debug(self._model.input_name)
        for layer in self._model.layers:
            LOG.debug(f"name: {layer['name']}, "
                      f"type: {layer['type']}, "
                      f"size: {layer['size']}")
        # FIXME[todo]: we have to create our TensorFlow layers from this
        # information.

        graph_def = self._graph.as_graph_def()
        LOG.debug(f"Operations in graph: {len(self._graph.get_operations())}, layers in model: {len(self._model.layers)}")
        scope = 'tmp'
        for i in range(len(self._model.layers)):
            name = scope + '/' + self._model.layers[i]['name'] + ':0'
            LOG.debug(name)
            tensor = self._graph.get_tensor_by_name(name)
            LOG.debug(f"{tensor.op.name} ({tensor.op.type}):")
            for op in tensor.consumers():
                LOG.debug(f"  -> {op.name} ({op.type})")

        #for n in graph_def.node:
        #    LOG.debug(f"Graph: {n.name}")
        # for op in graph.get_operations():
        #    LOG.debug(op.name)
        for lucid_layer in self._model.layers:
            LOG.debug(lucid_layer)
            operation = self._graph.get_operation_by_name(lucid_layer['name'])

        layer_dict = OrderedDict(layer_dict)
        return layer_dict


def load_lucid(name: str) -> Network:
    """Load the Lucid model with the given name.

    Returns
    -------
    model: LucidModel
        A reference to the LucidModel.
    """
    return Network(name=name)


def lucid_names() -> Iterator[str]:
    """Provide an iterator vor the available Lucid model names.

    Returns
    -------
    names: Iterator[str]
        An iterartor for the model names.
    """
    return nets_factory.models_map.keys()
