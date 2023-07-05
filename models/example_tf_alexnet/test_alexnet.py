"""Test the tensorflow AlexNet.

"""

# FIXME[todo]: this is not really a test suite yet. I have to
# design some additional tests.


# standard imports
from unittest import TestCase
import os

# third party imports
import tensorflow as tf
import numpy as np

# toolbox imports
from dltb.thirdparty.tensorflow.network import Network as TensorFlowNetwork
from dltb.util.image import imread, imresize

# from imagenet_classes import class_names


class TestAlexnet(TestCase):

    def setUp(self):
        self.checkpoint = os.path.join('models', 'example_tf_alexnet',
                                       'bvlc_alexnet.ckpt')
        # self.checkpoint = 'bvlc_alexnet.ckpt'
        # self.assertEqual(self.checkpoint, 'bvlc_alexnet.ckpt')

    def test_tensorflow_gpu(self):

        # if tf.test.gpu_device_name():
        #     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        # else:
        #     print("Please install GPU version of TF")
        #     print(tf.list_devices())
        self.assertTrue(tf.test.is_built_with_cuda())

    def test_tensorflow_network_(self):
        self.assertTrue(os.path.isfile(self.checkpoint + '.index'))

        network = TensorFlowNetwork(checkpoint=self.checkpoint)
        network.prepare()
        self.assertTrue(network.prepared)

        # print(network._sess.graph.get_operations())
        # network.summary()

        images = []
        for arg in ('images/elephant.jpg',):
            im = (imread(arg)[:, :, :3]).astype(np.float32)
            im = imresize(im, (227, 227))
            im = im - im.mean()
            im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
            images.append(im)

        self.assertEqual(len(images), 1)
        self.assertTrue(isinstance(images[0], np.ndarray))

        # Assuming the first op is the input.
        network_input_tensor = \
            network._session.graph.get_operations()[0].outputs[0]
        network_output_tensor = network['dense_3'].activation_tensor

        in_op = None
        out_op = None
        for op in network._session.graph.get_operations():
            # print(op.type)
            if op.type == 'Placeholder':
                _in_op = op
                print("Heureka: in!")
            if op.type == 'Softmax':
                out_op = op
                print("Heureka: out!")
                break

        if out_op:
            feed_dict = {network_input_tensor: images}
            output = network._session.run(network_output_tensor,
                                          feed_dict=feed_dict)

        for input_im_ind in range(output.shape[0]):
            inds = np.argsort(output)[input_im_ind, :]
            print("Image", input_im_ind)
            #for i in range(5):
            #    print("  {}: {} ({})".format(i, class_names[inds[-1-i]],
            #                                 output[input_im_ind, inds[-1-i]]))
