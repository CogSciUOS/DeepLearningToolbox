from typing import Iterator

from . import Network


def load_lucid(name: str) -> Network:
    """Load the Lucid model with the given name.

    Returns
    -------
    model: LucidModel
        A reference to the LucidModel.
    """

    from .lucid import Network as LucidNetwork
    network = LucidNetwork(name=name)
    return network


def lucid_names() -> Iterator[str]:
    """Provide an iterator vor the available Lucid model names.

    Returns
    -------
    names: Iterator[str]
        An iterartor for the model names.
    """
    import lucid.modelzoo.nets_factory as nets
    return nets.models_map.keys()
