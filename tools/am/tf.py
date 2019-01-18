"""The tools.am.tf module contains code to run activation maximization
on a Tensforflow network. This module is based on code developed by
Antonia Hain for her bachelor thesis. It was reworked be Ulf Krumnack
to fit into the qtpyvis toolbox.

It can be called directly from the command line as

  python -m tools.am.tf

or if for some reason 

  PYTHONPATH=$PWD:$PYTHONPATH python tools/am/tf.py

FIXME[old]: The functions 'createNetwork()' (together with the
auxiliary `conv()' function) seem to be obsolete. The code to create
the network has been moved to network.loader (function
'load_alexnet()').

FIXME[old]: Also the code in main() seems to be redundand with code in
tools.am.engine.
"""

import os
import numpy as np
import time
import cv2
import tensorflow as tf


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



def createNetwork(with_tensorboard:bool=False):
    train_x = np.zeros((1, 227, 227, 3)).astype(np.float32)
    train_y = np.zeros((1, 1000))
    
    xdim = train_x.shape[1:]
    ydim = train_y.shape[1]
    
    net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

    # I changed this to shape (1,)+xdim as gradient would be 0 on
    # undefined shapes so for now, you can only feed 1 image at a time
    x_in = tf.placeholder(dtype=tf.float32, shape=(1,) + xdim,  name='input_placeholder')

    if with_tensorboard:
        # plug image placeholder into tensorboard
        tf.summary.image("generated_image", x_in, 4)

    #conv1
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0], trainable=False)
    conv1b = tf.Variable(net_data["conv1"][1], trainable=False)
    conv1_in = conv(x_in, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    #maxpool1
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv2
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0], trainable=False)
    conv2b = tf.Variable(net_data["conv2"][1], trainable=False)
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    #lrn2
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)
    kernels = [(3,3),(5,5),(7,7)]

    #maxpool2
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0], trainable=False)
    conv3b = tf.Variable(net_data["conv3"][1], trainable=False)
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0], trainable=False)
    conv4b = tf.Variable(net_data["conv4"][1], trainable=False)
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    #conv5
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0], trainable=False)
    conv5b = tf.Variable(net_data["conv5"][1], trainable=False)
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    #maxpool5
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #fc6
    fc6W = tf.Variable(net_data["fc6"][0], trainable=False)
    fc6b = tf.Variable(net_data["fc6"][1], trainable=False)
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    #fc7
    fc7W = tf.Variable(net_data["fc7"][0], trainable=False)
    fc7b = tf.Variable(net_data["fc7"][1], trainable=False)
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    #fc8
    fc8W = tf.Variable(net_data["fc8"][0], trainable=False)
    fc8b = tf.Variable(net_data["fc8"][1], trainable=False)
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

    #prob
    prob = tf.nn.softmax(fc8)


    ## to make selection of layer of target unit easier. still requires
    ## knowledge of the network but oh well
    layer_dict = {
        'conv1' : conv1[0,10,10],
        'conv2' : conv2[0,10,10],
        'conv3' : conv3[0,10,10],
        'conv4' : conv4[0,10,10],
        'conv5' : conv5[0,0,0],
        'fc6'   : fc6[0],
        'fc7'   : fc7[0],
        'fc8'   : fc8[0],
        'prob'  : prob[0]
    }
    return x_in, layer_dict, prob


from network.tensorflow import Network as TensorFlowNetwork
from network.loader import load_alexnet

from tools.caffe_classes import class_names

#from tools import am_constants
from tools.am import Config

#from tools import am_tools
from tools.am import Engine


def main():
    config = Config()
    engine = Engine(config)
    
    # only slightly adapted by ahain

    # added: set summary folder, if summary is desired
    if config.TENSORBOARD_ACTIVATED:
        summary_dir = "summary/"
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

    #
    # NETWORK
    #
    #x_in, layer_dict, prob = createNetwork(config.TENSORBOARD_ACTIVATED)
  
    network = load_alexnet()
    engine.network = network
    
    #x_in = network._input_placeholder[0]
    x_in = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    prob = tf.get_default_graph().get_tensor_by_name('Softmax:0')

    print(type(x_in))
    print(type(prob), prob.name, prob, prob[0])


    # Doing this in a try except block to put out somewhat more
    # helpful errors.
    try:   
        # LAYER_KEY: the name of the layer in the layer_dictionary
        # current default: LAYER_KEY = 'fc8'
        #target_layer = layer_dict[config.LAYER_KEY]
        target_layer = tf.get_default_graph().get_tensor_by_name('xw_plus_b:0')[0]
        print(type(target_layer), target_layer)

        # selects random unit within layer boundaries or specific unit based
        # on configuration
        # FIXME[hack]:
        config._layer = target_layer
        config.random_unit()
        
        TO_MAXIMIZE = target_layer[config.UNIT_INDEX]
        idx = config.UNIT_INDEX

    # should happen when layer_dict[config.LAYER_KEY] fails
    except KeyError:
        raise KeyError("Your selected layer key seems to not exist. Please check in config.py")

    # should happen when config.UNIT_INDEX cannot be
    # accessed in the layer
    except ValueError:
        raise ValueError("Something went wrong selecting the unit to maximize. Have you made sure your UNIT_INDEX in config is within boundaries?")

    loss, grad = engine.createLoss(x_in, TO_MAXIMIZE)



    #
    # main part
    #

    t = time.time() # to meaure computation time

    with tf.Session() as sess:
        im, output, finish = engine.maximize_activation(sess, x_in, prob, loss, grad)

    print("\nComputation time", time.time()-t)


    #
    # Output
    #

    print("\nOutput following:\n")
    n = 6
    # output.shape[0] provides the batch size.
    # input_im_ind: index of input image
    for input_im_ind in range(output.shape[0]):

        indices = np.argsort(output)[input_im_ind,:]
        winneridx = np.argmax(output[input_im_ind])

        top_n = []
        for j in range(n):
            #print(class_names[indices[-1-j]], output[input_im_ind, indices[-1-j]])
            top_n.append(indices[-1-j])

        engine.outputClasses(input_im_ind, output[input_im_ind], idx, class_names, winneridx, top_n)
        engine.saveImage(im[0], idx, finish, top_n)


if __name__ == "__main__":
    main()
