
from observer import Observer, Observable, BaseChange, change
from network import Network
from model import Model

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


from typing import Iterator

class EngineChange(BaseChange):
    ATTRIBUTES = ['engine_changed', 'model_changed', 'unit_changed']


class EngineObserver(Observer):
    """An EngineObserver is notfied whenever some change in the state
    of the activation maximization Engine occurs.
    """

    def engineChanged(self, engine: 'Engine', info: EngineChange) -> None:
        """Respond to change in the activation maximization Engine.

        Parameters
        ----------
        engine: Engine
            Engine which changed (since we could observe multiple ones)
        info: ConfigChange
            Object for communicating which aspect of the engine changed.
        """
        pass


class Engine(Observable):
    """The Engine is a wrapper around the lucid module.

    Attributes
    ----------
    _model: LucidModel
        The currently selected lucid model. None if no model is
        selected.

    _model_name: str
        The name of the currently selected lucid model. None if
        no model is selected.
        
    _models_map: dict
        This is simply a reference to the
        lucid.modelzoo.nets_factory.models_map.
        It maps model name (`str`) to lucid models
        (lucid.modelzoo.vision_base.Model)

    """

    def __init__(self):
        super().__init__(EngineChange, 'engineChanged')
        self._models_map = nets.models_map
        self._model = None
        self._model_name = None
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
        return self._model_name

    @property
    def model_names(self) -> Iterator[str]:
        """Provide an iterator vor the available Lucid model names.

        Returns
        -------
        names: Iterator[str]
            An iterartor for the model names.
        """
        return self._models_map.keys()

    @change
    def load_model(self, name: str) -> LucidModel:
        """Load the Lucid model with the given name.

        Returns
        -------
        model: LucidModel
            A reference to the LucidModel.
        """

        try:
            factory = self._models_map[name]
            self._model = factory()
            self._model_name = name
            # load the graph definition (tf.GraphDef) from a binary
            # protobuf file and reset all devices in that GraphDef.
            self._model.load_graphdef()
        except:
            self._model = None
            self._model_name = None
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

        print("!!! running all:")
        for unit in range(self.layer_units):
            self.unit = unit
            self.notifyObservers(EngineChange(unit_changed=True))
            print(f"!!! running unit {unit}")
            obj = objectives.channel(self.layer_id, unit)
            self.image = render.render_vis(self.model, obj)
            if not self.running:
                break
            self._doRun(True)
            
        self._doRun(False)
