"""Integration of the torch library. Torch provides its
own array types, similar to numpy.

This module adds some hooks to work with pillow images:

* adapt :py:func:`dltb.base.image.Data` to allow transformation of
  `Datalike` objects to :py:class:`torch.Tensor` (as_torch) as well
  as and from :py:class:`PIL.Image.Image` to other formats.

* add a data kind 'torch' and an associated loader to the
  :py:class:`Datasource` class.

"""

# third party imports
import numpy as np
import torch

# toolbox imports
from datasource import Datasource
from ...base.data import Data, Datalike


def as_torch(data: Datalike, copy: bool = False) -> torch.Tensor:
    """Get a :py:class:`torch.Tensor` from a :py:class:`Datalike`
    object.
    """
    if isinstance(data, torch.Tensor):
        return data.clone().detach() if copy else data

    if isinstance(data, str) and data.endswith('.pt'):
        return torch.load(data)

    if isinstance(data, Data):
        if not hasattr(data, 'torch'):
            data.add_attribute('torch', Data.as_torch(data.array, copy=copy))
        return data.torch

    if not isinstance(data, np.ndarray):
        data = Data.as_array(data)

    # from_numpy() will use the same data as the numpy array, that
    # is changing the torch.Tensor will also change the numpy.ndarray.
    # On the other hand, torch.tensor() will always copy the data.
    return torch.tensor(data) if copy else torch.from_numpy(data)


print("adapting dltb.base.data.Data: adding static method 'as_torch'")
Data.as_torch = staticmethod(as_torch)

# add a loader for torch data: typical suffix is '.pt' (pytorch)
Datasource.add_loader('torch', torch.load)
