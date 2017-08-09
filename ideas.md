## Frameworks to integrate

The visualization toolbox should support different deep learning frameworks (sorted by priority):
  1. keras
  2. tensorflow
  1. keras with tf integration
    * https://www.tensorflow.org/api_docs/python/tf/contrib/keras
  3. caffe (http://caffe.berkeleyvision.org/)
  1. caffe2 https://caffe2.ai/
    * have a look at https://github.com/caffe2/caffe2/blob/afe1e7fabf62ab0c3ef44ca81115450213cbc907/caffe2/python/visualize.py
  4. deeplearning4j
    * https://deeplearning4j.org/
    * probably not usuable in life mode, since running on JVM
  5. theano
  1. pytorch (http://pytorch.org/)
  1. torch (http://torch.ch/)

see https://deeplearning4j.org/compare-dl4j-torch7-pylearn for an overview of deep learning frameworks

## Features to integrate

* Visualize the network structure
* Visualize patches that maximally activate neurons
  * occlusion
* Visualize the weights
* Visualize the representation space (t-SNE)
* Occlusion experiments
* Deconv approaches (single backward pass)
* Optimization over image approaches (optimization)
* Testing stability against adversarial images

## Other visualization approaches
* https://cs.stanford.edu/people/karpathy/convnetjs/  
* https://pair-code.github.io/deeplearnjs/demos/imagenet/imagenet-demo.html  
* https://github.com/PAIR-code/saliency

## References
* https://medium.com/@hint_fm/design-and-redesign-4ab77206cf9
* http://cs231n.stanford.edu/
* https://bcourses.berkeley.edu/courses/1453965
* http://lvdmaaten.github.io/tsne/
* http://scs.ryerson.ca/~aharley/vis/
* https://docs.google.com/presentation/d/1a-3bQwuc2Fjc1g8QeK6UuDzZOR1j1wvyWOxEyVWnxUc/edit#slide=id.g18174e0a77_0_1002
* https://github.com/jcjohnson/fast-neural-style
* https://icmlviz.github.io/reference/
* http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
* https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
* https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59
* http://yosinski.com/deepvis
* http://scs.ryerson.ca/~aharley/vis/conv/
* https://icmlviz.github.io/
* https://distill.pub/2016/misread-tsne/
* https://github.com/Evolving-AI-Lab/synthesizing
