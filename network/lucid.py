from collections import OrderedDict
from frozendict import FrozenOrderedDict

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# lucid.modelzoo.vision_models:
#     A module providinge the pretrained networks by name, e.g.
#     models.AlexNet
import lucid.modelzoo.vision_models as models
import lucid.modelzoo.nets_factory as nets
from lucid.modelzoo.vision_base import Model as LucidModel

from .tensorflow import Network as TensorflowNetwork



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

    def __init__(self, model: LucidModel=None, name: str=None, **kwargs):
        self._model = nets.models_map[name]() if model is None else model
        self._name = name
        logger.debug(f"New LucidNetwork({name}) {self._model}")
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

    def _create_layer_dict(self) -> FrozenOrderedDict:
        """Try to find the sequences in operations of the graph that
        match the idea of a layer.

        Returns
        -------
        A mapping of layer_ids to layer objects.
        """
        layer_dict = OrderedDict()

        # FIXME[hack]: create a layer dict
        # Here we can use the model.layers dictionary from the Lucid model.
        logger.debug(self._model.input_name)
        for layer in self._model.layers:
            logger.debug(f"name: {layer['name']}, "
                         f"type: {layer['type']}, "
                         f"size: {layer['size']}")
        # FIXME[todo]: we have to create our TensorFlow layers from this
        # information.

        graph_def = self._graph.as_graph_def()
        logger.debug(f"Operations in graph: {len(self._graph.get_operations())}, layers in model: {len(self._model.layers)}")
        scope = 'tmp'
        for i in range(len(self._model.layers)):
            name = scope + '/' + self._model.layers[i]['name'] + ':0'
            logger.debug(name)
            tensor = self._graph.get_tensor_by_name(name)
            logger.debug(f"{tensor.op.name} ({tensor.op.type}):")
            for op in tensor.consumers():
                logger.debug(f"  -> {op.name} ({op.type})")

        #for n in graph_def.node:
        #    logger.debug(f"Graph: {n.name}")
        # for op in graph.get_operations():
        #    logger.debug(op.name)
        for lucid_layer in self._model.layers:
            logger.debug(lucid_layer)
            operation = self._graph.get_operation_by_name(lucid_layer['name'])

        layer_dict = FrozenOrderedDict(layer_dict)
        return layer_dict
