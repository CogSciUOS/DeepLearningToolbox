"""Provide deep learning toolbox integration for torch.hub.

Torch.hub is a pre-trained model repository designed to facilitate
research reproducibility.


* `torch.hub.list(github, force_reload=False, skip_validation=False)`

  List all callable entrypoints available in the repo specified by github.

* `torch.hub.help(github, model, force_reload=False, skip_validation=False)`

  Show the docstring of entrypoint model.

* `torch.hub.load(repo_or_dir, model, *args, source='github',
    force_reload=False, verbose=True, skip_validation=False, **kwargs)`

  Load a model from a github repo or a local directory.

* `torch.hub.download_url_to_file(url, dst, hash_prefix=None, progress=True)`

  Download object at the given URL to a local path.

* `torch.hub.load_state_dict_from_url(url, model_dir=None, map_location=None,
     progress=True, check_hash=False, file_name=None)`

  Loads the Torch serialized object at the given URL.

* `torch.hub.get_dir()` and `torch.hub.set_dir(d)`

  Get and set the Torch Hub cache directory used for storing
  downloaded models & weights.

  The locations are used in the order of

    Calling hub.set_dir(<PATH_TO_HUB_DIR>)

    $TORCH_HOME/hub, if environment variable TORCH_HOME is set.

    $XDG_CACHE_HOME/torch/hub, if environment variable XDG_CACHE_HOME is set.

    ~/.cache/torch/hub

  By default, we don’t clean up files after loading it. Hub uses the
  cache by default if it already exists in the directory returned by
  get_dir().

  Users can force a reload by calling hub.load(...,
  force_reload=True). This will delete the existing github folder and
  downloaded weights, reinitialize a fresh download. This is useful
  when updates are published to the same branch, users can keep up
  with the latest release.

Issues
------

* The function `torch.hub._get_cache_or_reload` prints a line
  "Using cache found in ...", if a cached version of the repository
  directory is found.  This message can be turned off by passing
  the argument `verbose=False`.  However, the functions `torch.hub.list`
  and `torch.hub.help` have hardcoded the value `True` for this argument.
  when using `torch.hub.load`, the optional `verbose=False` argument
  can be provided, to turn this message off.


Examples
--------

from dltb.thirdparty.torch.hub import Network

network = Network('alexnet')


References
----------

[1] https://pytorch.org/hub/


Demo
----


from dltb.thirdparty.torch.network import DemoResnetNetwork
n = DemoResnetNetwork()
s = n.class_scores('examples/dog2.jpg')
print(s.argmax())  # 258
n.print_classification('examples/dog.jpg') # 258 (Samoyed) [0.8846]


-----
# Full code from which this class has been derived:

# https://pytorch.org/hub/pytorch_vision_resnet/

import torch
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
model.eval()


# Download an example image from the pytorch website
import urllib
#url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as
                                         # expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])

# The output has unnormalized scores. To get probabilities, you can
# run a softmax on it.
print(torch.nn.functional.softmax(output[0], dim=0))

print(output[0].argmax())

"""

# thirdparty imports
from typing import Iterable
import torch

# toolbox imports
from dltb.thirdparty.datasource.imagenet import ImagenetClassifier
from .network import ImageNetwork, ClassifierNetwork

# Default github to use
DEFAULT_GITHUB = 'pytorch/vision:v0.6.0'


class Network(ImageNetwork, ClassifierNetwork, ImagenetClassifier):
    """An network to be initialized from a torch hub pretrained model.

    Arguments
    ---------
    repository:
    """

    @staticmethod
    def models(github: str = DEFAULT_GITHUB) -> Iterable[str]:
        """Obtain available models from torch.hub
        """
        return torch.hub.list(github)

    def __init__(self, model=None, lookup: str = 'torch',
                 repository: str = DEFAULT_GITHUB,
                 **kwargs) -> None:
        super().__init__(model=model, lookup=lookup,
                         input_shape=(3, 224, 244),
                         # FIXME[hack]: input_shape can be different
                         **kwargs)
        self._repository = repository

    def _prepare(self) -> None:
        """Prepare the :py:class:`Network` by loading the model
        from the `torch.hub`.
        """
        if self._model is None:
            if not isinstance(self._model_init, str):
                raise TypeError("Illegal type for initializing a torch hub "
                                f"network model: {type(self._model_init)}")
            self._model = torch.hub.load(self._repository, self._model_init,
                                         pretrained=True, verbose=False)
        super()._prepare()


class DemoResnetNetwork(Network):
    """An experimental Torch Network based on ResNet.

    https://pytorch.org/hub/pytorch_vision_resnet/
    """

    def __init__(self, model='resnet18', **kwargs) -> None:
        super().__init__(model=model, **kwargs)
