"""Tools related to Networks.
"""
# standard imports
from typing import Tuple, Any

# toolbox imports
from ..tool import Tool, Context
from .network import Network

# FIXME[todo]: currently, this class seems not to be used anywhere ...
class NetworkTool(Tool):
    """A :py:class:`Tool` that utilizes a Network to perform its
    operation.

    """

    external_result: Tuple[str] = ('outputs', )
    internal_arguments: Tuple[str] = ('inputs_', )
    internal_result: Tuple[str] = ('outputs_', )

    @property
    def network(self) -> Network:
        """The network utilized by this `NetworkTool`.
        """
        return self._network

    @network.setter
    def network(self, network: Network) -> None:
        self._network = network

    def _preprocess(self, inputs, *args, **kwargs) -> Context:
        data = super()._preprocess(*args, **kwargs)
        if inputs is not None:
            data.add_attribute('_inputs', self._network.preprocess(inputs))
        return data

    def _process(self, inputs: Any, *args, **kwargs) -> Any:
        """Default operation: propagate data through the network.
        """
        output_layer = self._network.output_layer_id()
        return self._network.get_activations(inputs, output_layer)

    def _postprocess(self, context: Context, what: str) -> None:
        if what == 'outputs':
            context.add_attribute(what,
                                  self._network.postprocess(context.outputs_))
        elif not hasattr(context, what):
            raise ValueError(f"Unknown property '{what}' for tool {self}")
