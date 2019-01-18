from .engine import Engine, _EngineHelper, logger
from .config import Config

import os
import cv2
import time
import numpy as np
import tensorflow as tf

from network import Network


class _TensorflowHelper(_EngineHelper):
    """
    The TensorflowHelper is an implementation of an activation
    maximization engine for TensorFlow engines.

    Attributes
    ----------
    _layer: tensorflow.python.framework.ops.Tensor
        The layer to maximize, or that contains the unit to maximize.
        This is the layer identified by Config.LAYER_KEY.
        None, if no layer is selected (e.g., if Config.LAYER_KEY is
        invalid).

    _unit: tensorflow.python.framework.ops.Tensor
        The unit to maximize. This unit is identified by Config.UNIT.
        None, if no unit is selected.

    _loss: tensorflow.python.framework.ops.Tensor
        The loss to be computed.

    _grad: tensorflow.python.framework.ops.Tensor
        The gradient of the loss.

    _distances: numpy.ndarray
        A distance matrix for border punishment

    _session: FIXME[hack]
    _graph: FIXME[hack]
    """
    ## to make selection of layer of target unit easier. still requires
    ## knowledge of the network but oh well

    def __init__(self, engine: Engine,
                 config: Config=None, network: Network=None):
        super().__init__(engine, config, network)
        self._session = self._network._session
        self._graph = self._session.graph
        
        self._tf_layer = None
        self._unit = None

        self._loss = None
        self._grad = None

        # A distance matrix for border punishment
        self._distances = None

    def createLoss(self, input, unit):
        """Create a loss and gradient. The loss is the function to
        maximize a given unit. The created loss function may also
        include some regularization term.
      
        Parameters
        ----------
        input: tf.Tensor

        unit: tf.Tensor
        """
        n = sum(1 for _ in tf.get_default_graph().as_graph_def().node)
        logger.info(f"-TensorflowHelper.createLoss() -- begin"
                    f" ({self._config.L2_ACTIVATED}) nodes={n}, unit={type(unit)}")

        # add a placeholder for border regularizer (not used yet!)
        try:
            self._tf_unit = self._graph.get_tensor_by_name('Unit:0')
            logger.info(f"-TensorflowHelper.createLoss() -- reusing Unit")
        except KeyError:
            self._tf_unit = tf.placeholder(dtype=tf.int32, shape=(),
                                           name='Unit')
            logger.info(f"-TensorflowHelper.createLoss() -- created new Unit")

        try:
            self._tf_l2_activted = self._graph.get_tensor_by_name('L2_ACTIVATED:0')
            logger.info(f"-TensorflowHelper.createLoss() -- reusing L2_ACTIVATED")
        except KeyError:
            self._tf_l2_activted = tf.placeholder(dtype=tf.bool, shape=(),
                                                  name='L2_ACTIVATED')
            logger.info(f"-TensorflowHelper.createLoss() -- created new L2_ACTIVATED")

        try:
            self._tf_l2_lambda = self._graph.get_tensor_by_name('L2_LAMBDA:0')
            logger.info(f"-TensorflowHelper.createLoss() -- reusing L2_LAMBDA")
        except KeyError:
            self._tf_l2_lambda = tf.placeholder(dtype=tf.float32, shape=(),
                                                name='L2_LAMBDA')
            logger.info(f"-TensorflowHelper.createLoss() -- created new L2_LAMBDA")


        self._loss = -tf.reduce_sum(self._tf_layer[self._tf_unit])
        self._loss = tf.cond(self._tf_l2_activted,
                             lambda: self._loss +
                             self._tf_l2_lambda *
                             tf.reduce_sum(tf.multiply(input,input)),
                             lambda: self._loss)

        # old alternative: create a static loss path, based on the
        # current configuration. This does not allow to alter
        # configuration during maximzation.
        #
        # self._loss = -tf.reduce_sum(unit)
        # if self._config.L2_ACTIVATED:
        #    self._loss += (self._config.L2_LAMBDA *
        #                  tf.reduce_sum(tf.multiply(input,input)))

        #
        # regularize: borders
        #

        # an input for border regularizer
        try:
            self._tf_border_reg_activated = \
                self._graph.get_tensor_by_name('BORDER_REG_ACTIVATED:0')
            logger.info(f"-TensorflowHelper.createLoss() -- reusing BORDER_REG_ACTIVATED")
        except KeyError:
            self._tf_border_reg_activated = \
                tf.placeholder(dtype=tf.bool, shape=(),
                               name='BORDER_REG_ACTIVATED')
            logger.info(f"-TensorflowHelper.createLoss() -- created new BORDER_REG_ACTIVATED")

        try:
            self._tf_border_factor = self._graph.get_tensor_by_name('BORDER_FACTOR:0')
            logger.info(f"-TensorflowHelper.createLoss() -- reusing BORDER_FACTOR")
        except KeyError:
            self._tf_border_factor = tf.placeholder(dtype=tf.float32, shape=(),
                                                    name='BORDER_FACTOR')
            logger.info(f"-TensorflowHelper.createLoss() -- created new BORDER_FACTOR")


        try:
            self._center_distance = \
                self._graph.get_tensor_by_name('CenterDistance:0')
            logger.info(f"-TensorflowHelper.createLoss() -- reusing CenterDistance")
        except KeyError:
            input_shape = self._engine.get_image_shape(include_batch=False)
            self._center_distance = \
                tf.placeholder(dtype=tf.float32, shape=input_shape,
                               name='CenterDistance')
            logger.info(f"-TensorflowHelper.createLoss() -- created new CenterDistance")

        self._loss = tf.cond(self._tf_border_reg_activated,
                             lambda: self._loss +
                             tf.reduce_sum(tf.multiply(tf.abs(input),
                                                       self._center_distance)),
                             lambda: self._loss)

        #if self._config.BORDER_REG_ACTIVATED:
        #    self._loss = self.reg_border(input, self._center_distance,
        #                                 self._loss)
            # self._loss = self.reg_border_old(input, self._loss)

        # get gradient based on loss
        # todo: maybe normalize
        self._grad = tf.gradients([self._loss], [input])

        n = len([n.name for n in tf.get_default_graph().as_graph_def().node])
        logger.info(f"-TensorflowHelper.createLoss() -- end, nodes={n}")

        # more tensorboard
        if self._config.TENSORBOARD_ACTIVATED:
            tf.summary.scalar("loss", tf.reduce_mean(loss))
            self._summaries = tf.summary.merge_all()


    def reg_border_old(self, im_placeholder, loss):
        """Changes loss in order to punish values the closer to the
        border they are.
    
        Adds border region punishment by adding the product of the
        current image and the center distance image to the loss term.

        Arguments
        ---------
        im_placeholder:
            Placeholder for image in first network layer
        loss:
            loss tensor.

        Returns
        -------       
            loss modified such that it also includes the sum of all
            products of absolute pixel values on each color channel
            and their distance to the image center
        """
        x_dim = int(im_placeholder.shape[1])
        y_dim = int(im_placeholder.shape[2])

        # create array containing each coordinates' distance from
        # center with list comprehension
        distances = [[self._config.BORDER_FACTOR *
                      np.sqrt((x_dim/2-x)*(x_dim/2-x) +
                              (y_dim/2-y)*(y_dim/2-y)) for y in range(y_dim)]
                     for x in range(x_dim)]

        # not good code, I know, but so far the method I could come up
        # with fastest because tf-multiply doesn't seem to be able to
        # broadcast distances into (227,227,3) instead of (227,227)
        # automatically
        distances = np.stack((distances, distances, distances),axis=2)

        # add sum of all pixel value absolutes * their distance to
        # center to loss first I wanted to use the vector norm instead
        # of absolutes for each color channel but for some reason it
        # couldn't be derived and turned the gradients into nan
        return loss + tf.reduce_sum(tf.multiply(tf.abs(im_placeholder),
                                                distances))

    def reg_border(self, im_placeholder: tf.Tensor,
                   center_distance: tf.Tensor,
                   loss: tf.Tensor) -> tf.Tensor:
        """
        Adds border region punishment by adding the product of the
        current image and the center distance image to the loss term

        Arguments
        ---------
        im_placeholder:
            placeholder for optimized image
        center_distance:
            placeholder for distance image (/distance image crop when
            upscaling is used)
        loss:
            The loss tensor.
        
        Returns
        -------
            updated loss tensor
        """
        return loss + tf.reduce_sum(tf.multiply(tf.abs(im_placeholder),
                                                center_distance))

    # FIXME[todo]: make it async
    #  - check tensorflow sessions and Python threads
    #  - make the engine observable - store the generated images
    #  - develop the logging concept, remove print commands
    #  - show intermediate results and progress reports
    #  - allow for interactive control: pause and continue computation
    def prepare_maximization1(self, initialize_variables: bool):
        """Preparation of a call to perform_step, first phase.
        This part needs to be executed in the main thread.
        """
        t = time.time() # to meaure computation time
        logger.info(f"Preparation1 STEP 1: set input and output")
        if self._network:
            self._input = self._network.get_input_tensor()
            
            # LAYER_KEY: the name of the layer in the layer_dictionary
            # current default: LAYER_KEY = 'fc8'
            #target_layer = layer_dict[self._config.LAYER_KEY]
            #target_layer = tf.get_default_graph().get_tensor_by_name('xw_plus_b:0')[0]

            #self._tf_layer = self._graph.get_tensor_by_name('xw_plus_b:0')[0]
            self._tf_layer = self._network.get_output_tensor(pre_activation=True)
            if self._config and self._config.UNIT_INDEX is not None:
                self._unit = self._tf_layer[self._config.UNIT_INDEX]

            # The classification layer (actually not needed for maximization)
            #self._output = self._graph.get_tensor_by_name('Softmax:0')
            self._output = self._network.get_output_tensor()
          
            # Old:
            #self._tf_layer_dict = {
            #    'conv1' : conv1[0,10,10],
            #    'conv2' : conv2[0,10,10],
            #    'conv3' : conv3[0,10,10],
            #    'conv4' : conv4[0,10,10],
            #    'conv5' : conv5[0,0,0],
            #    'fc6'   : fc6[0],
            #    'fc7'   : fc7[0],
            #    'fc8'   : fc8[0],
            #    'prob'  : prob[0]
            #}
            # FIXME[hack]: this is a big hack!
            #x_in, layer_dict, prob = createNetwork()
            # x_in = network._input_placeholder[0]

        else:
            self._input = None
            self._output = None
            self._tf_layer = None
            self._unit = None
        logger.info(f"### layer: {type(self._tf_layer)}, {self._tf_layer}")
        logger.info(f"### config.UNIT_INDEX={self._config.UNIT_INDEX}, "
                    f"unit={self._unit} ###")
        logger.info(f"### output: {type(self._output)}, {self._output.name}, "
                    f"{self._output}, {self._output[0]}")
        logger.info(f" - Step1 time: {time.time()-t}")

        logger.info(f"Preparation1 STEP 2: Create loss")
        with self._graph.as_default():
            self.createLoss(self._input, self._unit)

        if initialize_variables:
            logger.info("Preparation1 STEP 3: initialize variables")
            self._session.run(tf.global_variables_initializer())
        
        logger.info(f"Preparation1 time: {time.time()-t}")

    def prepare_maximization2(self):
        """Preparation of a call to
        :py:meth:`EngineHelper.perform_step`, second phase.  This part
        can be executed in same thread as perform_step().
        """

        t = time.time()
        logger.info(f"-Preparation2 STEP 1: tensorboard {self._config.TENSORBOARD_ACTIVATED}")
        if self._config.TENSORBOARD_ACTIVATED:
            summary_dir = "summary/"
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            train_writer = tf.summary.FileWriter(summary_dir,
                                                 self._session.graph)

        #
        # Create a distance matrix
        #
        logger.info(f"-Preparation2 STEP 2: distance matrix for border punishment")

        # The distance image is used for border region punishment. As
        # it is now a placeholder this has to be included here, though
        # it may end up unused.
        image_shape = self._engine.get_image_shape(False,False)
        logger.info(f"-Preparation2 STEP 2: {image_shape}, {type(image_shape)}")
        coords = np.asarray(np.meshgrid(*[np.arange(d)
                                          for d in image_shape])).T
        center = np.asarray([image_shape[a]/2 for a in (1,0)])
        distances = (self._config.BORDER_FACTOR
                     * np.linalg.norm(coords - center, axis=2)
                     ** self._config.BORDER_EXP)
        # FIXME[question]: is that really necessary, can't we broadcast?
        self._distances = np.tile(distances[:,:,np.newaxis], (1,1,3))
        
        # FIXME[old]:
        #distances = [[self._config.BORDER_FACTOR *
        #              np.sqrt((image_shape[1]/2-x)**2 +
        #                      (image_shape[0]/2-y)**2) **
        #              self._config.BORDER_EXP
        #              for y in range(image_shape[0])]
        #             for x in range(image_shape[1])]
        #distances = np.stack((distances, distances, distances), axis=2)

        logger.info(f"-Preparation2: info")
        logger.info(f"--{self._config.LAYER_KEY}")
        logger.info(f"--Session: {self._session}")
        logger.info(f"--Graph: {self._graph}")
        logger.info(f"--Input: {self._input} ({type(self._input)})")
        logger.info(f"--Output: {self._output} ({type(self._output)})")
        logger.info(f"Preparation2 time: {time.time()-t}")

    def perform_step(self, image: np.ndarray, iteration: int):
        """Perform one optimization step. What exactly is done depends
        on the current :py:class:Config values.

        Paramters
        ---------
        image:
            The input image on which the activation maximization should
            be perfomed.
        iteration:
            The current iteration step. This is needed as some
            regularizations on occury every n-th step.
        
        Returns
        -------
        new_image: np.ndarray
            The updated image.
        loss: float
            The loss value for the original input image. 
        """
        c = self._config

        # get subimage for this step if transformation robustness is used
        use_subcoords = self._config.LARGER_IMAGE or self._config.JITTER
        if use_subcoords:
            # FIXME[orig]:
            # im, distances_crop, indx, indy, rand_x, rand_y = get_subimage(im_big, distances, xdim)
            # FIXME[todo]: do not recompute shape on every iteration!
            # FIXME[todo]: allow for larger batch sizes!
            image_shape = image.shape[1:]
            input_shape = self._engine.get_input_shape(include_batch=False)
            coords = self.get_subcoords(image_shape, input_shape)
            subimage = self.get_subimage(image, *coords, is_batch=True)
            subdistances = self.get_subimage(self._distances, *coords)
        else:
            subimage = image
            subdistances = self._distances

        #
        # Compute loss and gradients
        #
        
        # get new probability, loss, gradient potentially summary
        feed_dict = {self._input: subimage,
                     self._tf_unit: self._config.UNIT_INDEX,
                     self._tf_l2_activted: self._config.L2_ACTIVATED,
                     self._tf_l2_lambda: self._config.L2_LAMBDA,
                     self._tf_border_reg_activated: self._config.BORDER_REG_ACTIVATED,
                     self._tf_border_factor: self._config.BORDER_FACTOR,
                     self._center_distance: subdistances}
        if c.TENSORBOARD_ACTIVATED:
            loss, grad, summaries = \
                self._session.run([self._loss, self._grad, self._summaries],
                                  feed_dict=feed_dict)
            train_writer.add_summary(summaries, iteration)
        else:
            loss, grad = \
                self._session.run([self._loss, self._grad],
                                  feed_dict=feed_dict)

        # Update image according to gradient. doing this manually
        # because gradient was computed on placeholder, but is applied
        # to image
        # FIXME[design]: it would be nicer to let TensorFlow do the update ...
        subimage = subimage - c.ETA * np.asarray(grad[0])

        # regularize: clipping pixels with small contribution
        if (c.CONTRIBUTION_CLIPPING_ACTIVATED and
            iteration % c.CONTRIBUTION_CLIPPING_FREQUENCY == 0):
            subimage = self.reg_clip_contrib(subimage, grad)

        # plug sibimage back into big image
        if use_subcoords:
            self.set_subimage(image, subimage, *coords, is_batch=True)
        else:
            image = subimage
   
        # regularize: clipping pixels with small norm
        if (c.NORM_CLIPPING_ACTIVATED and
            iteration % c.NORM_CLIPPING_FREQUENCY == 0):
            image = self.reg_clip_norm(image)

        # regularize: gaussian blur
        if c.BLUR_ACTIVATED and iteration % c.BLUR_FREQUENCY == 0:
            image = self.reg_blur(image)

        return image, loss

    def network_output(self, input):
        return self._session.run(self._output,
                                 feed_dict = {self._input: input})

    def network_loss(self, input):
        feed_dict = {self._input: input,
                     self._tf_unit: self._config.UNIT_INDEX,
                     self._tf_l2_activted: self._config.L2_ACTIVATED,
                     self._tf_l2_lambda: self._config.L2_LAMBDA,
                     self._tf_border_reg_activated: self._config.BORDER_REG_ACTIVATED,
                     self._tf_border_factor: self._config.BORDER_FACTOR,
                     self._center_distance: distances_crop}
        # FIXME[todo]: determine distances_crop!
        return self._session.run(self._loss, feed_dict=feed_dict)

        # FIXME[old/todo]: there is some issue with larger images
        # here. Repair this!

        # get probability for complete (downsized) image for the
        # convergence criterion
        # one may remove this. it remains to be tested how much of
        # an actual difference it makes for convergence, but it seems
        # to be very costly
        if self._config.LARGER_IMAGE:
            # resized = np.zeros((1,227,227,3))
            input_shape = self._engine.get_input_shape()
            size = input_shape[1:3] # FIXME[hack]
            resized = np.zeros(input_shape)
            resized[0] = cv2.resize(im_big[0], size)
            _loss = sess.run(loss, feed_dict = {x_in:resized, center_distance: distances_crop})

    @property
    def layer(self):
        return self._config.LAYER_KEY, self._tf_layer
    
    @layer.setter
    def layer(self, layer: str):
        """
        Raises
        ------
        KeyError:
            Your selected layer (Config.LAYER_KEY) seems to not exist.
            FIXME[question]: there is now KeyError raised here, is it?
        """
        if layer != self._config.LAYER_KEY:
            self._config.LAYER_KEY = layer
        if layer is None:
            self._tf_layer = None
        elif self._network is not None:
            # FIXME[hack]: currently layer extraction does not work!
            # In future we may ues the layer_dict from the Network class!
            # self._tf_layer = layer_dict[self._config.LAYER_KEY]
            #t = tf.get_default_graph(). 
            t = self._graph.get_tensor_by_name('xw_plus_b:0')
            print("Tensor:", type(t), t)
            self._tf_layer = self._graph.get_tensor_by_name('xw_plus_b:0')[0]
            if self._config and self._config.UNIT_INDEX is not None:
                self._unit = self._tf_layer[self._config.UNIT_INDEX]

    @property
    def unit(self) -> (int, object):
        """Get the target unit.

        Result
        ------
        unit_index: int
            The index of the unit (in the current layer)
        unit: object
            The actual object representing the unit in the network.
        """
        print(f"self._config.UNIT_INDEX={self._config.UNIT_INDEX}, self._tf_layer={self._tf_layer}")
        return self._config.UNIT_INDEX, self._unit

    @unit.setter
    def unit(self, unit_index: int) -> None:
        """Update the unit to maximize.

        Arguments
        ---------
        index: int
            Index of the unit in the layer. A value of None means,
            that no single unit (but the entire layer) should be maximized.

        Raises
        ------
        ValueError:
            Something went wrong selecting the unit to maximize. Have
            you made sure your UNIT_INDEX in Config is within
            boundaries?
        """
        if unit_index != self._config.UNIT_INDEX:
            self._config.UNIT_INDEX = unit_index
        if unit_index is None:
            self._unit = None
        elif self._tf_layer is None:
            self._unit = None
        else:
            self._unit = self._tf_layer[unit_index]
