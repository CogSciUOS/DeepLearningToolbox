"""This module includes some methods frequently used in the AlexNet
activation maximization, such as regularization methods and a data
saving method.

Requires .config in order to work.
Antonia Hain, 2018

FIXME[todo]: wish list
 * stop/pause and continue optimization
 * remove extra nodes from graph
 * we could try to decouple the am stuff from the rest of the system
   so that it may be used standalone, i.e. make the Engine just
   a wrapper around some more stand alone activation maximization code.


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

from base.observer import Observable, change
from network import Network

from network.tensorflow import Network as TensorflowNetwork


# FIXME[hack]: make this network/dataset agnostic!
from tools.caffe_classes import class_names


class Engine(Observable, method='maximization_changed',
             changes=['engine_changed', 'config_changed', 'network_changed',
                      'image_changed']):
    """Activation Maximization Engine.

    The Engine takes all configuration values from the Config
    object. This will include the network name and the random
    seeds. Restarting the engine with the same Config object should
    yield identical results.


    Attributes
    ----------
    config: Config
        The engine will take all its parameters from the config object.        
    network: Network
        The :py:class:`Network` to be used for maximization.
    _image: np.ndarray
        The current image (can be batch of images), used for the
        maximization process.
    _snapshot: np.ndarray
        A snapshot (copy) of the current image (a singe image), can
        be used for display, etc.
    _running: bool
        A flag indicating if the engine is currently running.
    """
    _helper = None
    _running = False
    _status = "stopped"
    _network = None
    _iteration = None
    _image: np.ndarray = None
    _snapshot: np.ndarray = None

    def __init__(self, config: Config=None):
        super().__init__()
        self._config = config
        
        # Initialize the recorders
        self._recorder = {}
        for recorder in 'loss', 'min', 'max', 'mean', 'std':
            self._recorder[recorder] = None
        self._cache = {}

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Config) -> None:
        print(f"!!! MaximizationEngine: config={config} !!!")
        self._config.assign(config)

    @property
    def image(self):
        """Get the current snapshot.
        Usually a better way to get a copy of the snapshot is
        to call :py:meth:get_snapshot.
        """
        return self._snapshot

    @property
    def iteration(self):
        return self._iteration

    @property
    def status(self):
        return self._status

    @property
    def running(self):
        return self._running

    @property
    def description(self):
        return ("Artificial input generated to maximize "
                f"unit {self._config.UNIT_INDEX} "
                f"in layer {self._config.LAYER_KEY}")

    @change
    def _set_status(self, status, running=True):
        if status != self.status:
            self._status = status
            self._running = running
            # FIXME[hack]: we need a better notification concept!
            #self.change(engine_changed=True)
            self.notifyObservers(Engine.Change(engine_changed=True))

    @property
    def network(self) -> Network:
        """Get the network used by this maximization engine.
        """
        return self._network

    @network.setter
    def network(self, network: Network) -> None:
        """Assign a new network to this maximization engine.
        All future operations will be performed using this network.
        All initializations are lost by assigning a new network.
        :py:method:`_prepare` has to be run before the
        optimizer can be used (again).
        """
        print(f"!!! MaximizationEngine: network={network} !!!")
        if self._network == network:
            return  # nothing to do ...
    
        self._network = network
        self._helper = None
        if self._config is not None:
            self._config.NETWORK_KEY = network.get_id()

    @change
    def _prepare(self):
        """Prepare a new run of the activation maximization engine.
        This will adjust some parameters and make sure the Helper
        object for the current network is available, prepare the
        recorders and the loss list.
        """
        logger.info("-Engine._prepare() -- begin")
        self._set_status("preparation")

        # FIXME[hack]: determine these values automatically!
        # The axis along which the batches are arranged
        self._batch_axis = 0
        # The batch dimension (how many images per batch)
        self._batch_dimension = 1
        # The index of the image in the batch
        self._batch_index = 0

        if self._helper is None:
            HelperClass = _EngineHelper.class_for_network(self._network)
            self._helper = HelperClass(self, self._config, self._network)

        if self._helper is None:
            logger.warning("Engine: maximize activation: No Helper. STOP")
            return

        self._helper.prepare_maximization1(initialize_variables=False)

        #
        # Set up the recorders
        #
        for recorder in self._recorder.keys():
            self._init_recorder(recorder)

        #
        # sanity checks
        #

        # FIXME[design]: bad place to check this now. should be
        # checked by the config object.
        if self._config.LARGER_IMAGE and self._config.JITTER:
            raise Exception("Upscaling and jitter cannot be used simultaneously.")

        self._helper.prepare_maximization2()

        logger.debug("STEP 2: initialize image")
        self._image = self.initialize_image()
        logger.debug(f"STEP 2: image: {self._image.shape}")

        self._loss_list = np.ones(self._config.LOSS_COUNT) * -100000

        logger.info("-Engine._prepare() -- end")

    def _update_max_steps(self):
        """Change the maximal steps this engine is supposed to run.
        The new value is taken from the Config object, parameter MAX_STEPS.
        This essentially adapts the capacity of the recorders.
        """
        for recorder in self._recorder.values():
            if recorder is not None:
                recorder.resize((self._config.MAX_STEPS,) + recorder.shape[1:])

    def stop(self):
        """Stop the current optimization process. This will change
        the status to "stopped".
        """
        if self._helper is not None:
            self._set_status("stopped", False)
            logger.info("!!! Engine stopped !!!")

    def initialize_image(self) -> np.ndarray:
        """Provide an initial image for optimization.
        The creation takes the current network input size
        into account. The creation is controlled by the
        :py:class:`Config` parameter RANDOMIZE_INPUT: if True,
        a random image with values between -128 and 127 is
        created, otherwise the image is initialized with 0s.

        Returns
        -------
        image: np.ndarray
            The initial image (size is given by
            :py:meth:`_EngineHelper.get_image_shape`).
        """
        # FIXME[hack]: deal with large batch size and other batch axis
        batch_shape = (1,) + self._helper.get_image_shape(include_batch=False)
        if self._config.RANDOMIZE_INPUT:
            #image = np.random.randint(-127, 128, batch_shape)
            image = np.random.rand(*batch_shape) * 256 - 128
        else:
            image = np.full(batch_shape, 0, dtype=np.float)
        return image

    @change
    def maximize_activation(self, network: Network=None, reset: bool=True):
        """This is the actual optimization method: activation
        maximization.
        It will start an iterative optimization process performing
        gradient ascent, until one of the stop criteria is fulfilled.
        
        """
        logger.info("Engine: maximize activation: BEGIN")

        #
        # some sanity checks
        #
        if network is not None:
            self.network = network

        if self._network is None:
            raise RuntimeError("Cannot maximize activation without a Network.")

        if self._image is None or reset:
            # step counter
            self._iteration = -1
            self._set_status("initialization")
            self._prepare()
            # step counter

        #
        # main part
        #

        logger.debug(f"-Starting computation:")
        self._set_status("start")
        t = time.time()  # to meaure computation time
        self._set_status("running")

        #
        # get first loss
        #
        logger.debug("--STEP 1: get first loss (skipped)")
        loss = 0
        #loss = self._helper.network_loss(image)
        #logger.debug("Loss:", loss)
        # list containing last Config.LOSS_COUNT losses
        avg_steptime = 0  # step time counter

        # while current loss diverges from average of last losses by a
        # factor Config.LOSS_GOAL or more, continue, alternatively
        # stop if we took too many steps
        while (self._running
               and (not self._config.LOSS_STOP or
                    (np.abs(1-loss/np.mean(self._loss_list))
                    > self._config.LOSS_GOAL))
               and self._iteration < self._config.MAX_STEPS):

            # increase steps
            self._iteration += 1

            # start measuring time for this step
            start = time.time()

            # perform one optimization step
            self._image, loss = self._helper.perform_step(self._image,
                                                          self._iteration)

            # add previous loss to list. order of the losses doesn't
            # matter so just reassign the value that was assigned 
            # MAX_STEPS iterations ago
            self._loss_list[self._iteration % len(self._loss_list)] = loss

            # get time that was needed for this step
            avg_steptime += time.time() - start

            snapshot = self._image.take(self._batch_index,
                                        axis=self._batch_axis)
            self._take_snapshot(self._iteration, image=snapshot, loss=loss)
            self._record_snapshot()
            self.notifyObservers(Engine.Change(image_changed=True))

        self._set_status("stopped", False)

        # check if computation converged
        finish = (self._iteration < self._config.MAX_STEPS)

        # computation time
        comptime = time.time()-t
        # average step time
        avg_steptime /= self._iteration

        logger.debug(f"-TensorflowHelper.onMaximization() -- end")
        logger.debug(f"Computation time: {time.time()-t}")
        logger.debug(f"image: {self._image.shape}")

        #
        # store the result
        #
        self.finish = finish

        self.change(image_changed=True)
        logger.info(f"Engine.maximize_activation() -- end")

    def _take_snapshot(self, iteration: int, image: np.ndarray=None,
                       loss: float=None) -> None:
        """Take a snapshot of the engine state. Such a snapshot acts
        as short term memory to allow observers to access the state of
        the engine, while it already computes new values. It also acts
        as a cache to avoid recomputing values.
        """
        self._snapshot = { 'iteration': iteration }
        if image is not None:
            self._snapshot['image'] = image.copy()
        elif 'image' in self._recorder:
            self._snapshot['image'] = self._recorder['image'][iteration]
        if loss is not None:
            self._snapshot['loss'] = loss

    def _get_snapshot(self, what: str, iteration: int=None):
        """Get a snapshot of the activation process.
        A snapshot is a copy of the state of the process
        at a given iteration.

        Arguments
        ---------
        what: str
            A string describing which information to provide.
            Possible values are:
        iteration: int

        Returns
        ------
        The method will return None, is snapshots are not supported
        by this Engine.
        """
        if self._snapshot is None:
            return None

        if iteration is not None and self._snapshot['iteration'] != iteration:
            # create a new snapshot
            self._take_snapshot(iteration)

        if what in self._snapshot:
            # the snapshot already contains the desired information
            return self._snapshot[what]

        if (self._snapshot['iteration'] < self._iteration and
            what in self._recorder):
            # lookup the value from the recorder
            self._snapshot[what] = self._recorder[what][iteration]
            return self._snapshot[what]

        # we have to compute the requested value
        if 'image' not in self._snapshot:
            return None  # without image, we can not compute anything
        
        if what == 'min':
            self._snapshot[what] = self._snapshot['image'].min()
        elif what == 'max':
            self._snapshot[what] = self._snapshot['image'].max()
        elif what == 'mean':
            self._snapshot[what] = self._snapshot['image'].mean()
        elif what == 'std':
            self._snapshot[what] = self._snapshot['image'].std()
        elif what == 'normal':
            a = self._get_snapshot('max')
            b = self._get_snapshot('min')
            self._snapshot[what] = (0.5 if a==b else
                                    (self._snapshot['image'] - b)/(a-b))
        else:
            return None
        return self._snapshot[what]

    def get_snapshot(self, iteration: int=None, normalize: bool=False):
        return (self._get_snapshot('normal', iteration) if normalize else
                self._get_snapshot('image', iteration))

    def get_loss(self, iteration: int=None) -> float:
        return self._get_snapshot('loss', iteration)

    def get_min(self, iteration: int=None) -> float:
        return self._get_snapshot('min', iteration)

    def get_max(self, iteration: int=None) -> float:
        return self._get_snapshot('max', iteration)
    
    def get_mean(self, iteration: int=None) -> float:
        return self._get_snapshot('mean', iteration)

    def get_std(self, iteration: int=None) -> float:
        return self._get_snapshot('std', iteration)

    def _init_recorder(self, recorder):
        if recorder in self._recorder:
            shape = (self._config.MAX_STEPS,)
            if recorder == 'image':
                shape += self._helper.get_image_shape(include_batch=False)
            self._recorder[recorder] = np.ndarray(shape)

    def _record_snapshot(self):
        iteration = self._snapshot['iteration']
        for key in self._recorder.keys():
            self._recorder[key][iteration] = self._get_snapshot(key)

    def record(self, recorder, enabled: bool=True):
        if enabled and recorder not in self._recorder:
            self._recorder[recorder] = None
        elif not enabled and recorder in self._recorder:
            del self._recorder[recorder]

    def record_video(self, enabled: bool=True):
        """Activate or deactivate video recording.
        """
        self.record('image', enabled)

    def has_video(self):
        """Check if this engine provides an video.
        Video recording is deactivated by default, but can be turned on
        by calling :py:meth:record_video.
        """
        return 'image' in self._recorder

    def get_recorder_value(self, recorder, iteration=None, history=False):
        if iteration is None:
            iteration = self._iteration
        if iteration is None:
            return None
        if history:
            return self._recorder[recorder][:iteration]
        return self._recorder[recorder][iteration]

    def save_video(self, filename, fps=25):
        # http://www.fourcc.org/codecs.php
        # fourcc, suffix = 'PIM1', 'avi'
        # fourcc, suffix = 'ffds', 'mp4'
        fourcc, suffix = 'MJPG', 'avi' # Motion_JPEG
        # fourcc, suffix = 'XVID', 'avi'

        filename += '.' + suffix

        frames = self._recorder['image']
        frameSize = frames.shape[1:3]
        isColor = (frames.ndim==4)
        number_of_frames = self._iteration-1
        logger.info(f"save_video: preparing {filename} ({fourcc}) ... ")
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*fourcc),
                                 fps, frameSize, isColor=isColor)
        logger.info(f"save_video: writer.isOpened: {writer.isOpened()}")
        if writer.isOpened():
            logger.info(f"save_video: writing {number_of_frames} frames ... ")
            # FIXME[todo]: we need the number of actual frames!
            for frame in range(number_of_frames):
                writer.write(self._normalizeImage(frames[frame], as_bgr=True))
        logger.info(f"save_video: releasing the writer")
        writer.release()
        logger.info(f"save_video: done")

    def _normalizeImage(self, image: np.ndarray,
                        as_uint8:bool = True,
                        as_bgr = False) -> np.ndarray:
        # FIXME[design]: this put be done somewhere else ...
        normalized = np.ndarray(image.shape, image.dtype)
        cv2.normalize(image, normalized, 0, 255, cv2.NORM_MINMAX)
        # self.image_normalized = (self.image-min_value)*255/(max_value-min_value)
        if as_uint8:
            normalized = normalized.astype(np.uint8)
        if as_bgr:
            normalized = normalized[...,::-1]
        return normalized


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

    def get_input_shape(self, include_batch: bool=True) -> tuple:
        """The input shape is the shape in which the network
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
        return self._network.get_input_shape(include_batch)

    def get_image_shape(self, include_batch: bool=True,
                        include_colors: bool=True) -> tuple:
        """The image (data) shape is the shape of the actual
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
