"""NVlabs [1] is the label for NVIDIA research projects hosted on github.
These include some official implementations for research papers and
pretrained models.

Some implementations (e.g., stylegan, stylegan2) make use of `dnnlib`, a
deep learning module based on tensorflow (via `dnnlib.tflib`).


[1] https://github.com/NVlabs

"""


# The dnnlib.tflib interface
# ==========================



# TensorFlow dependencies
# -----------------------
#
# The package `dnnlib.tflib` officially requires TensorFlow version 1.x.
# However, in practice, it can be run with TensorFlow 2.x when activating
# the v1 compatibility mode and slightly adapting the source code:
# in `dnnlib.tflib.tfutil` the module `tensorflow.contrib` is imported,
# which does not exist in TensorFlow 2.0. However, it seems that this
# module is never really used, so the import may be commented out.


# Initializing TensorFlow
# -----------------------
#
# The function `init_tf` of the module `dnnlib.tflib.tfutil` initializes
# a default session if no default session exists yet (if there is already
# a default session, the function will do nothing).


from .stylegan import StyleGAN
from .stylegan2 import StyleGAN2
