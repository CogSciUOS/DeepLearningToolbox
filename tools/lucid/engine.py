import logging
logger = logging.getLogger(__name__)
print(f"!!!!!!!!!! getEffectiveLevel: {logger.getEffectiveLevel()} !!!!!!!!!!!!!")

from base.observer import Observable, change
from network import Network, loader
from network.lucid import Network as LucidNetwork

# lucid.modelzoo.vision_models:
#     A module providinge the pretrained networks by name, e.g.
#     models.AlexNet
import lucid.modelzoo.vision_models as models
import lucid.modelzoo.nets_factory as nets
from lucid.modelzoo.vision_base import Model as LucidModel

import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform


class Engine(Observable, method='engineChanged',
             changes=['engine_changed', 'model_changed', 'unit_changed']):
    """The Engine is a wrapper around the lucid module.

    Attributes
    ----------
    _network: LucidNetwork
        The currently selected lucid network. None if no model
        is selected.

    _model: LucidModel
        The currently selected lucid model. None if no model is
        selected.
        
    """

    def __init__(self):
        super().__init__()
        self._network = None
        self._model = None
        self._layer = None
        self._unit = None
        self.image = None
        self.running = False

    @property
    def model(self) -> LucidModel:
        """The currently selected lucid model. None if no model is
        selected.
        """
        return self._model

    @property
    def model_name(self) -> str:
        """The name of the currently selected lucid model. None if
        no model is selected.
        """
        return None if self._network is None else self._network.name

    @change
    def load_model(self, name: str) -> LucidModel:
        """Load the Lucid model with the given name.

        Returns
        -------
        model: LucidModel
            A reference to the LucidModel.
        """

        logger.info(f"load_model({name})")
        try:
            #self._network = LucidNetwork(name=name)
            self._network = loader.load_lucid(name)
            self._model = self._network.model
        except KeyError as e:
            self._network = None
            self._model = None
        logger.info(f"NAME={name}/{self.model_name} : {self._model}")
        self._layer = None
        self._unit = None
        self.change(model_changed=True, unit_changed=True)
        return self._model

    @change
    def set_layer(self, name: str, unit: int=0) -> None:
        """Set the currently selected layer.

        Arguments
        ---------
        name: str
            The name of the layer.
        unit: int
            The index of the unit in the layer.
        """
        if name == self.layer:
            return
        if self._model is None:
            return
        try:
            self._layer = next(x for x in self._model.layers
                               if x['name'] == name)
            self._unit = unit
        except StopIteration: # name not in layer list
            self._layer = None
            self._unit = None
        self.change(unit_changed=True)

    @property
    def layer(self) -> str:
        """The name of the currently selected layer.
        """
        return None if self._layer is None else self._layer['name']

    @layer.setter
    def layer(self, name: str) -> None:
        """Set the currently selected layer.
        """
        self.set_layer(name)

    @property
    def layer_type(self) -> str:
        """The type of the currently selected layer.
        """
        return None if self._layer is None else self._layer['type']

    @property
    def layer_units(self) -> int:
        """The number of units in the currently selected layer.
        """
        return None if self._layer is None else self._layer['size']

    @change
    def _set_unit(self, unit: int) -> None:
        if unit == self.unit:
            return
        if unit is None:
            self._unit = None
            self.change(unit_changed=True)
        elif self._layer is None:
            raise ValueError('Setting unit failed as no layer is selected')
        elif not 0 <= unit < self._layer['size']:
            raise ValueError(f"Invalid unit {unit} for current layer"
                             f" of size {self._layer['size']}")
        else:
            self._unit = unit
            self.change(unit_changed=True)

    @property
    def unit(self) -> int:
        """The index of the currently selected unit or None if no
        unit is selected.
        """
        return None if self._unit is None else self._unit

    @unit.setter
    def unit(self, unit: int) -> None:
        """The index of the currently selected unit or None if no
        unit is selected.
        """
        self._set_unit(unit)

    @property
    def layer_id(self) -> str:
        """The id of the currently selected layer or None if no
        unit is selected.
        """
        if self._layer is None:
            return None
        if self._layer['type'] == 'conv':
            return self._layer['name'] + '_pre_relu'
        return self._layer['name']

    @property
    def unit_id(self) -> str:
        """The id of the currently selected unit or None if no
        unit is selected.
        """
        return (None if self._layer is None
                else self.layer_id + ':' + str(self._unit))


    def _doRun(self, running: bool=True) -> None:
        self.running = running
        self.notifyObservers(EngineChange(engine_changed=True))

    def start(self):
        self.image = None
        self._doRun(True)

        obj = objectives.channel(self.layer_id, self.unit)
        self.image = render.render_vis(self.model, obj)
        #self.image = render.render_vis(self.model, self.unit_id)

        self._doRun(False)

    def stop(self):
        self._doRun(False)

    def start_multi(self):
        self.image = None
        self._doRun(True)

        logger.info("!!! running all:")
        for unit in range(self.layer_units):
            self.unit = unit
            self.notifyObservers(EngineChange(unit_changed=True))
            logger.info(f"!!! running unit {unit}")
            obj = objectives.channel(self.layer_id, unit)
            self.image = render.render_vis(self.model, obj)
            if not self.running:
                break
            self._doRun(True)
            
        self._doRun(False)

# FIXME[old]: this is too make old code happy. New code should use
# Engine.Change and Engine.Observer directly.
EngineChange = Engine.Change
EngineObserver = Engine.Observer
