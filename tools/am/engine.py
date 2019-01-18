"""This module includes some methods frequently used in the AlexNet
activation maximization, such as regularization methods and a data
saving method.

Requires .config in order to work.
Antonia Hain, 2018

FIXME[todo]: wish list
 * stop/pause and continue optimization
 * remove extra nodes from graph
 *

"""

from collections import OrderedDict

import os
import datetime
import time

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


import cv2
import numpy as np

from .config import Config

from observer import Observer, Observable, BaseChange, change
from network import Network
from model import Model

from network.tensorflow import Network as TensorflowNetwork


# FIXME[hack]: make this network/dataset agnostic!
from tools.caffe_classes import class_names


class EngineChange(BaseChange):
    ATTRIBUTES = ['engine_changed', 'config_changed', 'network_changed',
                  'image_changed']


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
    """Activation Maximization Engine.

    The Engine takes all configuration values from the Config
    object. This will include the network name and the random
    seeds. Restarting the engine with the same Config object should
    yield identical results.

    FIXME[todo]: There are some other aspects that we may work on in
    future:
      * we could try to decouple the am stuff from the rest of the system
        so that it may be used standalone, i.e. make the Engine just
        a wrapper around some more stand alone activation maximization code.

    Attributes
    ----------
    _config: Config
        The engine will take all its parameters from the config object.        
    _model: Model
        The model that allows to get access to networks (and maybe other
        data).

    _running: bool
        A flag indicating if the engine is currently running.
    """

    def __init__(self, model: Model=None, config: Config=None):
        super().__init__(EngineChange, 'engineChanged')
        self._config = config
        self._model = model
        self._helper = None
        self._running = False
        self._status = "stopped"

        self._iteration = -1
        # The current image (can be batch of images), used for the
        # maximization process
        self._image = None
        # A snapshot (copy) of the current image (a singe image), can
        # be used for display, etc.
        self._snapshot = None

        # FIXME[todo]: we need some recorder concept here ...
        self._loss = np.zeros(0)
        self._min = np.zeros(0)
        self._max = np.zeros(0)
        self._mean = np.zeros(0)
        self._std = np.zeros(0)
        self._images = None

    @property
    def image(self):
        return self._snapshot

    @property
    def iteration(self):
        return self._iteration

    # FIXME[concept]: as Config is observable, the question arises
    # what to do if the config of an engine is changed. I see three
    # possible ways:
    #  (1) do not care about Observers [this is what currently happens]
    #  (2) move the Observers from the old Config to the new Config.
    #  (3) do not actually set the Config but just copy its values
    #      (so the Observers will be informed about new values).
    # The last (3) seems the cleanest solution.
    @change
    def _set_config(self, config: Config) -> None:
        self._config = config
        self.change(config_changed=True)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Config) -> None:
        self._set_config(config)

    def get_image_shape(self, include_batch: bool=True,
                        include_colors: bool=True) -> tuple:
        """The image (data) shape.  This is the shape of the actual
        image created during the optimization process. It should be at
        least input_shape, but may be larger if the 'larger image'
        technique is used.

        Parameters
        ----------
        include_batch: bool
            A flag indicating if the shape tuple should include the
            batch axis.
        include_colors: bool
            A flag indicating if the shape tuple should include the
            color axis (if available).

        Returns
        -------
        shape: tuple
            The shape of the image.
        """
        image_shape = self.get_input_shape(include_batch)

        # FIXME[hack]: assuming (batch,dim_x,dim_y,colors)
        # This may actually be different, depending on the network.
        if self._config.LARGER_IMAGE:
            image_shape = (image_shape[:1] +
                           (self._config.IMAGE_SIZE_X,
                            self._config.IMAGE_SIZE_Y) +
                           image_shape[3:])

        if not include_colors:
            image_shape = image_shape[:-1]

        return image_shape


    def get_input_shape(self, include_batch: bool=True) -> tuple:
        """The input shape.  That is the shape in which the network
        expects its data, i.e., the shape of the input layer of the
        network.

        Parameters
        ----------
        include_batch:
            A flag indicating if the shape tuple should include the
            batch axis.

        Returns
        -------
        shape: tuple
            The shape of the input data.
        """
        network = self._model.network_by_id(self._config.NETWORK_KEY)
        return network.get_input_shape(include_batch)




    #
    # Ouput result and save data
    #
    def save_am_data(outputim, idx, finish, steps, comptime, avg_steptime, top5, impath, paramfilename):
        """
        Saves the activation maximization result data (image and
        parameter values).  The image is normalized or not depending
        on NORMALIZE_OUTPUT in Config.  Image and data will be
        saved with a timestamp in order to be able to match the data
        to one another more easily.  If the parameter file was not
        found, a new csv file including headline will be created at
        the specified location.

        Arguments
        ---------
        outputim: the image to save
        finish: indicating whether maximization process was finished or aborted
                due to too many steps being taken
        top5: top 5 object classes of outputim as determined by network
        impath: path where to save the image (not including filename)
        paramfilename: reference to file where parameters are saved
        """

        # get current time and condense to small id
        # ex. 24th of July, 19:28:15 will be 2407_192815
        timestamp = datetime.datetime.now().strftime("%d%m_%H%M%S")

        # normalize image if desired
        if self._config.NORMALIZE_OUTPUT:
            cv2.normalize(outputim,  outputim, 0, 255, cv2.NORM_MINMAX)


        filename = f"{idx}_{timestamp}.png"

        cv2.imwrite(impath+filename, outputim)
        # write parameters to file with or without headline depending
        # on if file was found
        if not os.path.isfile(paramfilename):
            with open(paramfilename,"a") as param_file:
                print("Specified parameter output file not found. Preparing file including headline.")

                param_file.write("timestamp,ETA,RANDOMIZE_INPUT,BLUR_ACTIVATED,BLUR_KERNEL,BLUR_SIGMA,BLUR_FREQUENCY,L2_ACTIVATED,L2_LAMBDA,NORM_CLIPPING_ACTIVATED,NORM_CLIPPING_FREQUENCY,NORM_PERCENTILE,CONTRIBUTION_CLIPPING_ACTIVATED,CONTRIBUTION_CLIPPING_FREQUENCY,CONTRIBUTION_PERCENTILE,BORDER_REG_ACTIVATED,BORDER_FACTOR,BORDER_EXP, JITTER_ACTIVATED,JITTER_STRENGTH,LARGER_IMAGE,LOSS_GOAL,MAX_STEPS,finish,steps,comptime, average time per step\n")

                param_file.write(f"{timestamp},{self._config.ETA},{self._config.RANDOMIZE_INPUT},{self._config.BLUR_ACTIVATED},{self._config.BLUR_KERNEL[0]}x{self._config.BLUR_KERNEL[1]},{self._config.BLUR_SIGMA},{self._config.BLUR_FREQUENCY},{self._config.L2_ACTIVATED},{self._config.L2_LAMBDA},{self._config.NORM_CLIPPING_ACTIVATED},{self._config.NORM_CLIPPING_FREQUENCY},{self._config.NORM_PERCENTILE},{self._config.CONTRIBUTION_CLIPPING_ACTIVATED},{self._config.CONTRIBUTION_CLIPPING_FREQUENCY},{self._config.CONTRIBUTION_PERCENTILE},{self._config.BORDER_REG_ACTIVATED},{self._config.BORDER_FACTOR},{self._config.BORDER_EXP},{self._config.JITTER},{self._config.JITTER_STRENGTH},{self._config.LARGER_IMAGE},{self._config.LOSS_GOAL},{self._config.MAX_STEPS},{finish},{steps},{comptime},{avg_steptime}\n")
        else:
            with open(paramfilename,"a") as param_file:
                param_file.write(f"{timestamp},{self._config.ETA},{self._config.RANDOMIZE_INPUT},{self._config.BLUR_ACTIVATED},{self._config.BLUR_KERNEL[0]}x{self._config.BLUR_KERNEL[1]},{self._config.BLUR_SIGMA},{self._config.BLUR_FREQUENCY},{self._config.L2_ACTIVATED},{self._config.L2_LAMBDA},{self._config.NORM_CLIPPING_ACTIVATED},{self._config.NORM_CLIPPING_FREQUENCY},{self._config.NORM_PERCENTILE},{self._config.CONTRIBUTION_CLIPPING_ACTIVATED},{self._config.CONTRIBUTION_CLIPPING_FREQUENCY},{self._config.CONTRIBUTION_PERCENTILE},{self._config.BORDER_REG_ACTIVATED},{self._config.BORDER_FACTOR},{self._config.BORDER_EXP},{self._config.JITTER},{self._config.JITTER_STRENGTH},{self._config.LARGER_IMAGE},{self._config.LOSS_GOAL},{self._config.MAX_STEPS},{finish},{steps},{comptime},{avg_steptime}\n")

    def save_am_data_old(self, image, idx, finish, top5, impath, paramfilename):
        """Saves the activation maximization result data (image and
        parameter values).  The image is normalized or not depending
        on NORMALIZE_OUTPUT in self._config.  Image and data will be
        saved with a timestamp in order to be able to match the data
        to one another more easily.  If the parameter file was not
        found, a new csv file including headline will be created at
        the specified location.

        Arguments
        ---------
        image:
            the image to save
        finish:
            indicating whether maximization process was finished or aborted
            due to too many steps being taken
        top5:
            top 5 object classes of image as determined by network
        impath:
            path where to save the image (not including filename)
        paramfilename:
            reference to file where parameters are saved
        """

        # get current time and condense to small id
        # ex. 24th of July, 19:28:15 will be 2407_192815
        timestamp = datetime.datetime.now().strftime("%d%m_%H%M%S")

        # OpenCV changes the order of the color channels!
        cv2image = image[:,:,::-1].copy()
        # normalize image if desired
        if self._config.NORMALIZE_OUTPUT:
            cv2.normalize(cv2image, cv2image, 0, 255, cv2.NORM_MINMAX)

        # plug together filename (extra line for readability)
        filename = (f"{timestamp}_{self._config.LAYER_KEY}_{idx}"
                    f"_classified_{top5}.png")

        cv2.imwrite(os.path.join(impath,filename), cv2image)

        output = OrderedDict([
            ('timestamp', f"{timestamp}"),
            ('NETWORK_KEY', f"{self._config.NETWORK_KEY}"),
            ('LAYER_KEY', f"{self._config.LAYER_KEY}"),
            ('UNIT_INDEX', f"{self._config.UNIT_INDEX}"),
            ('ETA', f"{self._config.ETA}"),
            ('RANDOMIZE_INPUT', f"{self._config.RANDOMIZE_INPUT}"),
            ('BLUR_ACTIVATED', f"{self._config.BLUR_ACTIVATED}"),
            ('BLUR_KERNEL',
             f"{self._config.BLUR_KERNEL[0]}x{self._config.BLUR_KERNEL[1]}"),
            ('BLUR_SIGMA', f"{self._config.BLUR_SIGMA}"),
            ('BLUR_FREQUENCY', f"{self._config.BLUR_FREQUENCY}"),
            ('L2_ACTIVATED', f"{self._config.L2_ACTIVATED}"),
            ('L2_LAMBDA', f"{self._config.L2_LAMBDA}"),
            ('NORM_CLIPPING_ACTIVATED',
             f"{self._config.NORM_CLIPPING_ACTIVATED}"),
            ('NORM_CLIPPING_FREQUENCY',
             f"{self._config.NORM_CLIPPING_FREQUENCY}"),
            ('NORM_PERCENTILE', f"{self._config.NORM_PERCENTILE}"),
            ('CONTRIBUTION_CLIPPING_ACTIVATED',
             f"{self._config.CONTRIBUTION_CLIPPING_ACTIVATED}"),
            ('CONTRIBUTION_CLIPPING_FREQUENCY',
             f"{self._config.CONTRIBUTION_CLIPPING_FREQUENCY}"),
            ('CONTRIBUTION_PERCENTILE',
             f"{self._config.CONTRIBUTION_PERCENTILE}"),
            ('BORDER_REG_ACTIVATED', f"{self._config.BORDER_REG_ACTIVATED}"),
            ('BORDER_FACTOR', f"{self._config.BORDER_FACTOR}"),
            ('LOSS_GOAL', f"{self._config.LOSS_GOAL}"),
            ('MAX_STEPS', f"{self._config.MAX_STEPS}"),
            ('finish', f"{finish}")])

        # write parameters to file with or without headline depending
        # on if file was found
        if not os.path.isfile(paramfilename):
            with open(paramfilename,"w") as param_file:
                print("Specified parameter output file "
                      f"'{paramfilename}' not found. "
                      "Preparing file including headline.")
                param_file.write(",".join(output.keys()) + "\n")
                param_file.write(",".join(output.values()) + "\n")
        else:
            with open(paramfilename,"a") as param_file:
                param_file.write(",".join(output.values()) + "\n")

    def get_top_n(self, output, n=5, class_names=None):
        """
        This was taken and minimally adapted from the original AlexNet class
        prints top 5 object classes and saves them in a list as well
        """
        indices = np.argsort(output)
        top_n = list(indices[-1:-n:-1])
        if class_names is None:
            return top_n
        else:
            return top_n, list(class_names[top_n])

    def outputClasses(self, input_im_ind, output, idx, class_names, winneridx, top_n_indices):
        """Get the output class.
        This was taken and minimally adapted from the original AlexNet class
        prints top 5 object classes and saves them in a list as well
        """

        print(f"Image {input_im_ind} generated to maximize unit {idx}"
              f" ({class_names[idx]})")

        # ADDED THIS MYSELF
        print(f"Winner is {class_names[winneridx]} at index {winneridx} with prob {output[winneridx]}\n")

        for j, index in enumerate(top_n_indices):
            print(j+1, index, class_names[index], output[index])
        print()


    def saveImage(self, outputim, idx, finish, top5):
        """Save data.
        """

        ## using these paths on my system, change or comment out on other computer
        # self.save_am_data(outputim, idx, finish, top5,
        #     impath=r"//media/antonia/TOSHIBA EXT/BA Result Images/",
        #     paramfilename="//media/antonia/TOSHIBA EXT/BA Result Images/parameters.csv")

        # works on every system: just save in current folder
        return self.save_am_data(outputim, idx, finish, top5, impath="",
                                 paramfilename="parameters.csv")


    @property
    def status(self):
        return self._status

    @property
    def running(self):
        return self._running

    @change
    def _set_status(self, status, running=True):
        if status != self.status:
            self._status = status
            self._running = running
            # FIXME[hack]: we need a better notification concept!
            #self.change(engine_changed=True)
            self.notifyObservers(EngineChange(engine_changed=True))

    @change
    def prepare(self):
        """Prepare a new run of the activation maximization engine.
        """
        logger.info("-Engine.prepare() -- begin")
        self._set_status("preparation")

        network = self._model.network_by_id(self._config.NETWORK_KEY)

        if self._helper is None:
            HelperClass = _EngineHelper.class_for_network(network)
            self._helper = HelperClass(self, self._config, network)
            self._helper.prepare_maximization1(initialize_variables=False)
        else:
            self._helper.prepare_maximization1(initialize_variables=False)

        self._loss = np.zeros(self._config.MAX_STEPS)
        if self._min is not None:
            self._min = np.zeros(self._config.MAX_STEPS)
        if self._max is not None:
            self._max = np.zeros(self._config.MAX_STEPS)
        if self._mean is not None:
            self._mean = np.zeros(self._config.MAX_STEPS)
        if self._std is not None:
            self._std = np.zeros(self._config.MAX_STEPS)
        if self._images is not None:
            self._images = None # FIXME!

        self._sanity_check()
        self._helper.prepare_maximization2()


        logger.debug("STEP 2: initialize image")
        self._image = self.initialize_image()
        logger.debug(f"STEP 2: image: {self._image.shape}")

        self._loss_list = np.ones(self._config.LOSS_COUNT) * -100000

        logger.info("-Engine.prepare() -- end")

    def stop(self):
        if self._helper is not None:
            self._set_status("stopped", False)
            logger.info("!!! Engine stopped !!!")

    def initialize_image(self) -> np.ndarray:
        """Provide an initial image for optimization.

        Returns
        -------
        image: np.ndarray
            The initial image (size is given by
            :py:meth:`Engine.get_image_shape`).
        """
        # FIXME[hack]: deal with large batch size and other batch axis
        batch_shape = (1,) + self.get_image_shape(include_batch=False)
        if self._config.RANDOMIZE_INPUT:
            #image = np.random.randint(-127, 128, batch_shape)
            image = np.random.rand(*batch_shape) * 256 - 128
        else:
            image = np.full(batch_shape, 0, dtype=np.float)
        return image


    # FIXME[question]: what is the idea of this function?
    def _sanity_check(self):
        
        if self._helper is None:
            logger.warning("Engine: maximize activation: No Helper. STOP")
            return

        # FIXME[hack]: should not check private variable
        #if self._config._layer is None:
        #    return  # Nothing to do

        # FIXME[hack]: should only be selected on explicit demand!
        #if self._config.UNIT_INDEX is None:
        #    self._config.random_unit()

        # FIXME[design]: bad place to check this now. should be
        # checked by the config object.
        if self._config.LARGER_IMAGE and self._config.JITTER:
            raise Exception("Upscaling and jitter cannot be used simultaneously.")


    @change
    def maximize_activation(self, reset: bool=True):
        """This is the actual maximization method.
        
        """
        logger.info("Engine: maximize activation: BEGIN")

        # FIXME[hack]: determine these values automatically!x
        # The axis along which the batches are arranged
        batch_axis = 0
        # The batch dimension
        batch_dimension = 1
        # The index of the image in the batch
        batch_index = 0

        if self._image is None or reset:
            # step counter
            self._iteration = -1
            self._set_status("initialization")
            self.prepare()
            # step counter
            self._iteration = 0
            self._set_status("start")

        #
        # main part
        #

        logger.debug(f"-Starting computation:")
        t = time.time() # to meaure computation time
        self._set_status("running")

        #
        # get first loss
        #
        logger.debug("--STEP 1: get first loss (skipped)")
        loss = 0
        #loss = self._helper.network_loss(image)
        #logger.debug("Loss:", loss)
        # list containing last Config.LOSS_COUNT losses
        avg_steptime = 0 # step time counter

        # while current loss diverges from average of last losses by a
        # factor Config.LOSS_GOAL or more, continue, alternatively
        # stop if we took too many steps
        while (self._running
               and (np.abs(1-loss/np.mean(self._loss_list))
                    > self._config.LOSS_GOAL)
               and self._iteration < self._config.MAX_STEPS):

            # start measuring time for this step
            start = time.time()

            # perform one optimization step
            self._image, loss = self._helper.perform_step(self._image,
                                                          self._iteration)

            # add previous loss to list. order of the losses doesn't
            # matter so just reassign the value that was assigned 50
            # iterations ago
            self._loss_list[self._iteration % len(self._loss_list)] = loss

            # get time that was needed for this step
            avg_steptime += time.time() - start


            # record history
            self._loss[self._iteration] = loss

            self._snapshot = self._image.take(batch_index,
                                              axis=batch_axis).copy()
            if self._min is not None:
                self._min[self._iteration] = self._snapshot.min()
            if self._max is not None:
                self._max[self._iteration] = self._snapshot.max()
            if self._mean is not None:
                self._mean[self._iteration] = self._snapshot.mean()
            if self._std is not None:
                self._std[self._iteration] = self._snapshot.std()

            # increase steps
            self._iteration += 1
            self.notifyObservers(EngineChange(image_changed=True))

        self._set_status("stopped", False)

        # check if computation converged
        finish = (self._iteration < self._config.MAX_STEPS)

        # computation time
        comptime = time.time()-t
        # average step time
        avg_steptime /= self._iteration

        logger.debug(f"-TensorflowHelper.onMaximization() -- end")
        logger.debug(f"Computation time: {time.time()-t}")
        logger.debug(f"image: {self.image.shape}")

        #
        # store the result
        #
        self.finish = finish

        self.change(image_changed=True)
        logger.info(f"Engine.maximize_activation() -- end")


    @property
    def description(self):
        return ("Artificial input generated to maximize "
                f"unit {self._config.UNIT_INDEX} "
                f"in layer {self._config.LAYER_KEY}")

        
    def old(self):

        #
        # Output
        #

        logger.debug("Classification results following:")
        activation = self._helper.network_output(image)
        self.activation = activation
        n = 6
        for batch_index in range(activation.shape[batch_axis]):
            dist = activation[batch_index]

            top_n = self.get_top_n(dist, n)

            logger.debug(f"Image {batch_index+1} of {activation.shape[batch_axis]}"
                         f" generated to maximize unit {self._config.UNIT_INDEX}"
                         f" ({class_names[self._config.UNIT_INDEX]})")
            logger.debug(f"Winner is '{class_names[top_n[0]]}' at index {top_n[0]}"
                         f" with prob {dist[top_n[0]]}. Top {n} following:")
            for j, index in enumerate(top_n):
                logger.debug(f" {j+1}. {class_names[index]} ({index}): "
                             f"{dist[index]*100}%")

            #
            # Save the resulting image
            #
            impath = ""
            paramfilename = "parameters.csv"
            self.save_am_data(self._snapshot, self._config.UNIT_INDEX,
                              finish, top_n,
                              impath=impath,
                              paramfilename=paramfilename)




class _EngineHelper:
    """:py:class:_EngineHelper is an auxiliary class realizing network
    related aspects of activation maximzation. :py:class:_EngineHelper
    is an abstract base class from which framework specific
    subclasses, like :py:class:_TensorflowHelper are derived.

    The :py:class:_EngineHelper is considered a private class and
    subject to change. No external code should rely on the API of this
    class or even its mere existence.

    Currently, this class provides all regularization methods, not
    only the network specific aspects. The idea is that
    the :py:class:Engine can use these methods to perform the actual
    maximization process.

    The central method is :py:meth:perform_step can 
    provide the activation maximization gradient for a given input,
    taking the current :py:class:Config values into account.

    Attributes
    ----------
    _engine: Engine
        A reference to the :py:class:Engine to be supported by this
        :py:class:_EngineHelper.

    _config: Config
        A :py:class:Config object providing configuration values for
        the maximization process.

    _network: Network
        The :py:class:Network used for activation maximization.   
    """

    @classmethod
    def class_for_network(cls, network):
        if isinstance(network, TensorflowNetwork):
            from .tensorflow import _TensorflowHelper as _EngineHelperClass
            return _EngineHelperClass
        else:
            raise NotImplementedError(f"Networks of type {type(network)}"
                                      " are currently not supported. Sorry!")

    def __init__(self, engine: Engine,
                 config: Config=None, network: Network=None):
        """Initialize this engine helper.

        Parameters
        ----------
        engine:
            The :py:class:Engine to be supported by this
            :py:class:_EngineHelper.
        config:
            A :py:class:Config object providing configuration values for
            the maximization process.
        network:
            The :py:class:Network used for activation maximization.   
        """
        self._engine = engine
        self._config = config
        self._network = network

    def get_subcoords(self, shape: tuple, subshape: tuple):
        """Get coordinates for a subimage. This can be used for
        upscaling or jitter.

        Arguments
        ---------
        shape: tuple
            Size of the original image.
        subshape: tuple
            Size of the subimage.

        Returns
        -------
        indx, indy: slice
            Indices that describe the part of the big image to be used.
            In simple cases, this may be a crop, but it may also apply
            a wrap around.
        rand_x, rand_y:
            the positions from which the part was cropped
        """
        if self._config.LARGER_IMAGE:
            # we allow the subimage to be partly (or even fully)
            # outside the image
            rand_x = np.random.randint(-subshape[1], shape[1])
            rand_y = np.random.randint(-subshape[0], shape[0])
        if self._config.JITTER:
            rand_x = np.random.randint(-self._config.JITTER_STRENGTH,
                                       self._config.JITTER_STRENGTH)
            rand_y = np.random.randint(-self._config.JITTER_STRENGTH,
                                       self._config.JITTER_STRENGTH)

        # FIXME[incomplete]: ok, this works for Config.WRAP_AROUND,
        # but what to do, when we run outside the image?
        indx = np.arange(rand_x, rand_x+subshape[1]) % shape[1]
        indy = np.arange(rand_y, rand_y+subshape[0]) % shape[0]

        return (indx, indy)


    def get_subimage(self, image: np.ndarray,
                     indx: np.ndarray, indy: np.ndarray,
                     is_batch: bool=False) -> np.ndarray:
        """Get a subimage of the given image. This can be used for
        upscaling or jitter.

        Arguments
        ---------
        image: np.ndarray
            full image
        indx, indy: slice
            Indices that describe the part of the big image to be used.
            In simple cases, this may be a crop, but it may also apply
            a wrap around.

        Returns
        -------
        subimage: np.ndarray
            cropped image part
        """

        x,y = np.meshgrid(indx, indy)
        if is_batch:
            subimage = image[:,y,x]
        else:
            subimage = image[y,x]

        return subimage

    def set_subimage(self, image: np.ndarray, subimage: np.ndarray,
                     indx: np.ndarray, indy: np.ndarray,
                     is_batch:bool=False) -> None:
        """Replace a part of a (larger) image by a given subimage.
        This can be used for upscaling or jitter.

        Arguments
        ---------
        image: np.ndarray
            The large image.
        subimage: np.ndarray
            The smaller image to put into the large image.
        indx, indy: slice
            Indices that describe the part of the larger image
            to be replaced. The number of indices should match the
            width and height of the submimage.
        """
        i = np.meshgrid(indx, indy)
        if is_batch:
            image[:,i[1],i[0]] = subimage
        else:
            image[i[1],i[0]] = subimage

    # old: can be removed after once we have checked that the new
    # implementation covers all functionality ...
    def set_subimage_old(image, subimage, indx, indy, rand_x, rand_y):
        """
        Better description needed. Plugs section back into the image

        Arguments
        ---------
        image:
        subimage:
        indx:
        indy:
        rand_x:
        rand_y:

        Returns
        -------
        image:

        """

        # in case pixel indices beyond the border are supposed to be
        # filled in on the other side of the original image
        if self._config.WRAP_AROUND:
            # get temporary image section of right composition (this
            # might be two sections of image stitched together if the
            # selected section crossed the image border)
            tmpsection = image[:,indx]
            tmpsection[:,:,indy] = subimage # put updated section in temporary image
            image[:,indx] = tmpsection # plug into big image

        # no wrap-around
        # pixels across borders were optimized, but are not to be
        # included in big image again
        else:

            sectionindy = indy
            sectionindx = indx

            # if section crossed image border
            if 0 in indy:
                # get index of 0 to determine where the border was
                # crossed
                wrap_index_y = indy.index(0)
                # to avoid cases where the subimage just randomly
                # started at 0
                if wrap_index_y > 0:
                    # cases where the subimage starts somewhere in the
                    # image and crosses the right border
                    if rand_y > 0:
                        # slice indices to be kept (start to right
                        # border)
                        indy = indy[:wrap_index_y]
                        # get corresponding indices for the subimage
                        # (necessary when the result image is
                        # upscaled)
                        sectionindy = range(len(indy))
                    # cases where the subimage starts from beyond the
                    # left big image border
                    else:
                        # slice indices to be kept (starting from left border)
                        indy = indy[wrap_index_y:]
                        # get corresponding indices for the subimage
                        sectionindy = range(subimage.shape[2]-len(indy),
                                            subimage.shape[2])

                # update temporary image section with the pixels to be kept
                tmpsection = subimage[:,:,sectionindy]
            else:
                # no need to worry about boundaries if the border was
                # not crossed
                tmpsection = subimage

            # similar to y dimension
            if 0 in indx:
                wrap_index_x = indx.index(0)
                if wrap_index_x > 0:
                    if rand_x > 0:
                        indx = indx[:wrap_index_x]
                        sectionindx = range(len(indx))
                    else:
                        indx = indx[wrap_index_x:]
                        sectionindx = range(subimage.shape[1]-len(indx),
                                            subimage.shape[1])
                tmpsection2 = tmpsection[:,sectionindx]
            else:
                tmpsection2 = tmpsection

        # plug updated, cut section into big image
        image[:,indx[0]:indx[0]+tmpsection2.shape[1],
               indy[0]:indy[0]+tmpsection2.shape[2]] = tmpsection2

        return image

    def reg_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian blur on the image according to given constants in
        self._config.

        Arguments
        ---------
        image: np.ndarray
            The image. It shouldn't matter if the image itself is 2- or 3d
            but it should be 1x(image_dimensions) as this is the network
            placeholder shape
            
            FIXME[concept]: we should decide how to deal with batches here:
            either blur full batch, or just hand over an image, not a batch

            FIXME[concept]: we should decide how to return the blurred
            image: either in place (as it is done now), or in another
            array. Have a look at how numpy/opencv does this.

        Returns
        -------
        blurred: np.ndarray
            The blurred image.
        """

        image[0] = cv2.GaussianBlur(image[0],
                                    self._config.BLUR_KERNEL,
                                    self._config.BLUR_SIGMA)
        return image

    def reg_clip_norm(self, image: np.ndarray):
        """Clips pixels with small norm in image (= sets them to 0)
        according to given constant in self._config.

        There are two methods of doing this (put the ones you don't
        want to use in comments.) Not yet entirely sure which of the
        two methods is the correct one as it isn't really indicated in
        the paper but according to the code in the deep visualization
        toolbox, I think the currently enabled one should be
        fine. Doesn't make much of a difference in practice.

        Arguments
        ---------
        image: np.ndarray
            The image. It shouldn't matter if the image itself is 2- or 3d
            but it should be 1x(image_dimensions) as this is the network
            placeholder shape
            
            FIXME[concept]: we should decide how to deal with batches here:
            either blur full batch, or just hand over an image, not a batch

            FIXME[concept]: we should decide how to return the blurred
            image: either in place (as it is done now), or in another
            array. Have a look at how numpy/opencv does this.

        Returns
        -------
        clipped: np.ndarray
            Image with small norm pixels clipped.
        """

        # get pixel norms
        norms = np.linalg.norm(image[0],axis=2)
        method = 1
        if method == 1:
            # method 1: clip all below percentile value in norms as
            # computed by np percentile
            norm_threshold = np.percentile(norms, self._config.NORM_PERCENTILE)

            # leaving this for completeness but I think it's wrong
            # norm_threshold = np.mean(norms) * NORM_PERCENTILE/100
        else:
            # method 2: clip all below value of pixel at
            # NORM_PERCENTILE of sorted norms
            norms_sorted = np.argsort(norms.flatten())
        
            # gets index of norm at PERCENTILE in the sorted array
            norm_index = norms_sorted[int(len(norms_sorted) *
                                          self._config.NORM_PERCENTILE/100)]
            # gets actual norm value to use as threshold
            norm_threshold = norms[np.unravel_index(norm_index,norms.shape)]

        # create mask and clip
        mask = norms < norm_threshold
        image[0][mask] = 0
        return image

    def reg_clip_contrib(self, image: np.ndarray, gradients: np.ndarray):
        """Clips pixels with small contribution in image (= sets them
        to 0) according to given constant in self._config.

        There are two methods of doing this (put the ones you don't
        want to use in comments.) See reg_clip_norm for more details
        on this issue.

        Arguments
        ---------
        image: np.ndarray
            The image. It shouldn't matter if the image itself is 2- or 3d
            but it should be 1x(image_dimensions) as this is the network
            placeholder shape

        gradients: np.ndarray
            Gradients computed by the network.

        Returns
        -------
        new_image: np.ndarray
            Image with small contribution pixels clipped.
        """
        # contribs = np.sum(image * self._config.ETA *
        #                   np.asarray(gradients[0]), axis=3)

        contribs = np.sum(image * np.asarray(gradients[0]), axis=3)
        # I tried using the absolute values but it didn't work well:
        # contribs = np.sum(np.abs(image) * np.asarray(gradients[0]), axis=3)

        method = 1
        if method == 1:
            # method 1: clip all below percentile value in contribs as
            # computed by np percentile
            contrib_threshold = np.percentile(contribs, self._config.
                                              CONTRIBUTION_PERCENTILE)
        else:
            # method 2: clip all below value of pixel at
            # CONTRIBUTION_PERCENTILE of sorted contribs
            contribs_sorted = np.argsort(contribs.flatten())
            contrib_index = contribs_sorted[int(len(contribs_sorted) *
                                                self._config.
                                                CONTRIBUTION_PERCENTILE/100)]
            contrib_threshold = contribs[np.unravel_index(contrib_index,
                                                          contribs.shape)]

        # create mask and clip
        mask = contribs < contrib_threshold
        image[mask] = 0
        return image


