# PYTHONPATH=$PWD:$PYTHONPATH python3 -m contrib.styletransfer
"""Based on code by Lucas Feldmann provided for the seminar
"Deep Learning for Computer Vision" (summer 2020).
"""

# standard imports
from argparse import ArgumentParser

# third party imports
import imageio
import tensorflow as tf

# toolbox imports
from dltb.base.image import ImageDisplay
#from models.styletransfer import StyletransferData
from qtgui.panels.styletransfer import StyletransferData
from models.styletransfer import StyletransferTool


def main():
    """Start the program."""

    parser = ArgumentParser(description='Style Transfer Demo Script')

    parser.add_argument('--init', help='Image to use for initialization'
                        ' (zeros, random, content, style, FILE)')
    # FIXME[bug]: the value 'style' seems not to be valid. See comments in
    # StyletransferTool for details

    args = parser.parse_args()

    # assert tf.__version__ >= '2.0.0'
    tf.compat.v1.enable_eager_execution()

    print(f"TensorFlow: is_gpu_available: {tf.test.is_gpu_available()}")
    # cuda_only=False, min_cuda_compute_capability=None

    data = StyletransferData()
    data.prepare()

    content_layers = ['block5_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    tool = StyletransferTool(style_layers=style_layers,
                             content_layers=content_layers)
    tool.content = imageio.imread(data._content['venice'])
    tool.style = imageio.imread(data._styles['starry_night'])
    tool.reset(args.init)

    display = ImageDisplay()  # loop=True
    display.run(tool)

    # FIXME[bug]: sometimes this raises an internal error (during the
    # invocation of function tf.linalg.einsum in function gram_matrix):
    #   Blas xGEMM launch failed : a.shape=[1,147456,64],
    #   b.shape=[1,147456,64], m=64, n=64, k=147456 [Op:Einsum]
    # I have not yet figured out what causes this error.


if __name__ == '__main__':
    main()
