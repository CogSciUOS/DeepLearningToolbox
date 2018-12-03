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

import cv2
import numpy as np

from .config import Config

from observer import Observer, Observable, BaseChange, change
from network import Network
from model import Model

from network.tensorflow import Network as TensorflowNetwork

# FIXME[bad]: we should not need tensorflow in the general part
import tensorflow as tf

# FIXME[debug]
import threading

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
        self._logger = lambda x: None
        self._logger2 = lambda x,y,z: None
        self._running = False

        self.image = None
        self.activation = None
        self.finish = None

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

    def print_logger(self, message):
        print(f"[[{message}]]")

    def print_logger2(self, image, iteration, loss):
        print(f"iteration {iteration}: loss={loss}")

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        self._logger = logger

    @property
    def logger2(self):
        return self._logger2

    @logger2.setter
    def logger2(self, logger):
        self._logger2 = logger


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


    def prepare(self):
        self._logger("-Engine.prepare() -- begin")

        network = self._model.network_by_id(self._config.NETWORK_KEY)

        if isinstance(network, TensorflowNetwork):
            if self._helper is None:
                self._helper = TensorflowHelper(self, self._config, network,
                                                self._logger)
                self._helper.prepare_maximization1(initialize_variables=False)
            else:
                self._helper.prepare_maximization1(initialize_variables=False)
        else:
            self._helper = None
            raise NotImplementedError(f"Networks of type {type(network)}"
                                      " are currently not supported. Sorry!")

        self._logger("-Engine.prepare() -- end")

    def stop(self):
        if self._helper is not None:
            self._running = False # FIXME[hack]
            self._logger("!!! Engine stopped !!!")


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


    def _sanity_check(self):
        if self._helper is None:
            self._logger("Engine: maximize activation: No Helper. STOP")
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
    def maximize_activation(self):
        """This is the actual maximization method.
        
        """
        self._logger("Engine: maximize activation: BEGIN")

        self._sanity_check()
        self._helper.prepare_maximization2()

        # FIXME[hack]: determine these values automatically!x
        # The axis along which the batches are arranged
        batch_axis = 0
        # The batch dimension
        batch_dimension = 1
        # The index of the image in the batch
        batch_index = 0

        self._logger("STEP 2: initialize image")
        image = self.initialize_image()
        self._logger(f"STEP 2: image: {image.shape}")

        #
        # main part
        #

        self._logger(f"-Starting computation:")
        t = time.time() # to meaure computation time
        self._running = True

        #
        # get first loss
        #
        self._logger("--STEP 1: get first loss (skipped)")
        loss = 0
        #loss = self._helper.network_loss(image)
        #self._logger("Loss:", loss)
        # list containing last Config.LOSS_COUNT losses
        avg_steptime = 0 # step time counter
        loss_list = [-100000 for _ in range(self._config.LOSS_COUNT)]

        # step counter
        i = 0 

        # while current loss diverges from average of last losses by a
        # factor Config.LOSS_GOAL or more, continue, alternatively
        # stop if we took too many steps
        while (self._running
               and np.abs(1-loss/np.mean(loss_list)) > self._config.LOSS_GOAL
               and i < self._config.MAX_STEPS):

            # start measuring time for this step
            start = time.time()

            # perform one optimization step
            image, loss = self._helper.perform_step(image, i)

            # add previous loss to list. order of the losses doesn't
            # matter so just reassign the value that was assigned 50
            # iterations ago
            loss_list[i % 50] = loss

            # increase steps
            i += 1

            # get time that was needed for this step
            avg_steptime += time.time() - start

            if self._logger2 is not None:
                self._logger2(image.take(batch_index, axis=batch_axis),
                              i, loss)

        self._running = False

        # check if computation converged
        finish = (i < self._config.MAX_STEPS)

        # computation time
        comptime = time.time()-t
        # average step time
        avg_steptime /= i

        self._logger(f"-TensorflowHelper.onMaximization() -- end")
        self._logger(f"Computation time: {time.time()-t}")
        self._logger(f"image: {image.shape}")

        #
        # store the result
        #
        self.image = image.copy()
        self.finish = finish
        self.description = ("Artificial input generated to maximize "
                            f"unit {self._config.UNIT_INDEX} in layer ")

        self.change(image_changed=True)
        self._logger(f"Engine.maximize_activation() -- end")

    def old(self):

        #
        # Output
        #

        self._logger("Classification results following:")
        activation = self._helper.network_output(image)
        self.activation = activation
        n = 6
        for batch_index in range(activation.shape[batch_axis]):
            dist = activation[batch_index]

            top_n = self.get_top_n(dist, n)

            self._logger(f"Image {batch_index+1} of {activation.shape[batch_axis]}"
                         f" generated to maximize unit {self._config.UNIT_INDEX}"
                         f" ({class_names[self._config.UNIT_INDEX]})")
            self._logger(f"Winner is '{class_names[top_n[0]]}' at index {top_n[0]}"
                         f" with prob {dist[top_n[0]]}. Top {n} following:")
            for j, index in enumerate(top_n):
                self._logger(f" {j+1}. {class_names[index]} ({index}): "
                             f"{dist[index]*100}%")

            #
            # Save the resulting image
            #
            impath = ""
            paramfilename = "parameters.csv"
            self.save_am_data(self.image, self._config.UNIT_INDEX,
                              finish, top_n,
                              impath=impath,
                              paramfilename=paramfilename)







class EngineHelper:
    """

    Attributes
    ----------
    _engine: Engine
    _config: Config
    _network: Network
    _logger:
    
    """

    def __init__(self, engine: Engine,
                 config: Config=None, network: Network=None,
                 logger=lambda x: None):
        """Initialize this engine helper.

        Parameters
        ----------
        engine:
        config:
        network:
        logger:
        """
        self._engine = engine
        self._config = config
        self._network = network
        self._logger = logger

    def get_subcoords(self, shape: tuple, subshape: tuple):
        """
        Get coordinates for a subimage. This can be used for
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
        """
        Get a subimage of the given image. This can be used for
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

        i = np.meshgrid(indx, indy)
        if is_batch:
            #tmp = image[:,indy]
            #subimage = tmp[:,:,indx]
            subimage = image[:,i[1],i[0]]
        else:
            #tmp = image[indy]
            #subimage = tmp[:,indx]
            subimage = image[i[1],i[0]]

        return subimage

    def set_subimage(self, image: np.ndarray, subimage: np.ndarray,
                     indx: np.ndarray, indy: np.ndarray,
                     is_batch:bool=False) -> None:
        i = np.meshgrid(indx, indy)
        if is_batch:
            image[:,i[1],i[0]] = subimage
        else:
            image[i[1],i[0]] = subimage

    
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



class TensorflowHelper(EngineHelper):
    """
    The TensorflowHelper is an implementation of an activation
    maximization engine for TensorFlow engines.

    Attributes
    ----------
    _session: FIXME[hack]
    _graph: FIXME[hack]
    _layer: tensorflow.python.framework.ops.Tensor
        The layer to maximize, or that contains the unit to maximize.
        This is the layer identified by Config.LAYER_KEY.
        None, if no layer is selected (e.g., if Config.LAYER_KEY is
        invalid).
    _unit: tensorflow.python.framework.ops.Tensor
        The unit to maximize. This unit is identified by Config.UNIT.
        None, if no unit is selected.
    """
    ## to make selection of layer of target unit easier. still requires
    ## knowledge of the network but oh well

    def __init__(self, engine: Engine,
                 config: Config=None, network: Network=None,
                 logger=lambda x: None):
        super().__init__(engine, config, network, logger)
        self._session = self._network._sess
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
        
        The function is currently implemented for TensorFlow engines.
        
        Parameters
        ----------
        input: tf.Tensor

        unit: tf.Tensor
        """

        # boolean for convenience
        trans_robust = self._config.LARGER_IMAGE or self._config.JITTER

        n = len([n.name for n in tf.get_default_graph().as_graph_def().node])
        self._logger(f"-TensorflowHelper.createLoss() -- begin"
                     f" ({self._config.L2_ACTIVATED}) nodes={n}, unit={type(unit)}")

        # an input for border regularizer
        try:
            self._tf_unit = self._graph.get_tensor_by_name('Unit:0')
            self._logger(f"-TensorflowHelper.createLoss() -- reusing Unit")
        except KeyError:
            self._tf_unit = tf.placeholder(dtype=tf.int32, shape=[1],
                                           name='Unit')
            self._logger(f"-TensorflowHelper.createLoss() -- created new Unit")

        ## define loss (potentially including L2 loss) and gradient
        ## loss is negative activation of the unit
        ## + (if l2 activated) lambda * image^2
        self._loss = (-tf.reduce_sum(unit)
                      + self._config.L2_ACTIVATED
                      * self._config.L2_LAMBDA
                      * tf.reduce_sum(tf.multiply(input,input)))

        #self._loss2 = (-tf.reduce_sum(self._tf_layer[self._tf_unit])
        #              + self._config.L2_ACTIVATED
        #              * self._config.L2_LAMBDA
        #              * tf.reduce_sum(tf.multiply(input,input)))


        # alternative:
        # self._loss = -tf.reduce_sum(unit)
        # if self._config.L2_ACTIVATED:
        #    self._loss += (self._config.L2_LAMBDA *
        #                  tf.reduce_sum(tf.multiply(input,input)))

        #
        # regularize: borders
        #

        # an input for border regularizer
        try:
            self._center_distance = \
                self._graph.get_tensor_by_name('CenterDistance:0')
            self._logger(f"-TensorflowHelper.createLoss() -- reusing CenterDistance")
        except KeyError:
            input_shape = self._engine.get_image_shape(include_batch=False)
            self._center_distance = \
                tf.placeholder(dtype=tf.float32, shape=input_shape,
                               name='CenterDistance')
            self._logger(f"-TensorflowHelper.createLoss() -- created new CenterDistance")

        if self._config.BORDER_REG_ACTIVATED:
            self._loss = self.reg_border(input, self._center_distance,
                                         self._loss)
            # self._loss = self.reg_border_old(input, self._loss)

        ## get gradient based on loss
        ## todo: maybe normalize
        self._grad = tf.gradients([self._loss], [input])

        n = len([n.name for n in tf.get_default_graph().as_graph_def().node])
        self._logger(f"-TensorflowHelper.createLoss() -- end, nodes={n}")

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
        self._logger(f"Preparation1 STEP 1: set input and output")
        if self._network:
            # FIXME[hack]!
            # x_in
            self._input = self._graph.get_tensor_by_name('Placeholder:0')
            # prob
            self._tf_layer = self._graph.get_tensor_by_name('xw_plus_b:0')[0]
            if self._config and self._config.UNIT_INDEX is not None:
                self._unit = self._tf_layer[self._config.UNIT_INDEX]
            # The classification layer (actually not needed for maximization)
            self._output = self._graph.get_tensor_by_name('Softmax:0')

            # LAYER_KEY: the name of the layer in the layer_dictionary
            # current default: LAYER_KEY = 'fc8'
            #target_layer = layer_dict[self._config.LAYER_KEY]
            # FIXME[async]: this will throw a KeyError:
            # "The name 'xw_plus_b:0' refers to a Tensor which does not exist. The operation, 'xw_plus_b', does not exist in the graph."
            # Seemingly the network is not available in another thread.
            #target_layer = tf.get_default_graph().get_tensor_by_name('xw_plus_b:0')[0]
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
        self._logger(f"### layer: {type(self._tf_layer)}, {self._tf_layer}")
        self._logger(f"### config.UNIT_INDEX={self._config.UNIT_INDEX}, "
                     f"unit={self._unit} ###")
        self._logger(f"### output: {type(self._output)}, {self._output.name}, "
                     f"{self._output}, {self._output[0]}")
        self._logger(f" - Step1 time: {time.time()-t}")

        self._logger(f"Preparation1 STEP 2: Create loss")
        self.createLoss(self._input, self._unit)

        if initialize_variables:
            self._logger("Preparation1 STEP 3: initialize variables")
            self._session.run(tf.global_variables_initializer())
        
        self._logger(f"Preparation1 time: {time.time()-t}")

    def prepare_maximization2(self):
        """Preparation of a call to
        :py:meth:`EngineHelper.perform_step`, second phase.  This part
        can be executed in same thread as perform_step().
        """

        t = time.time()
        self._logger(f"-Preparation2 STEP 1: tensorboard {self._config.TENSORBOARD_ACTIVATED}")
        if self._config.TENSORBOARD_ACTIVATED:
            summary_dir = "summary/"
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            train_writer = tf.summary.FileWriter(summary_dir,
                                                 self._session.graph)

        #
        # Create a distance matrix
        #
        self._logger(f"-Preparation2 STEP 2: distance matrix for border punishment")

        # The distance image is used for border region punishment. As
        # it is now a placeholder this has to be included here, though
        # it may end up unused.
        image_shape = self._engine.get_image_shape(False,False)
        self._logger(f"-Preparation2 STEP 2: {image_shape}, {type(image_shape)}")
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

        self._logger(f"-Preparation2: info")
        self._logger(f"--{self._config.LAYER_KEY}")
        self._logger(f"--Session: {self._session}")
        self._logger(f"--Graph: {self._graph}")
        self._logger(f"--Input: {self._input} ({type(self._input)})")
        self._logger(f"--Output: {self._output} ({type(self._output)})")
        self._logger(f"Preparation2 time: {time.time()-t}")

    def perform_step(self, image: np.ndarray, iteration: int):
        """Perform one optimization step. What exactly is done depends
        on the settings in the Config object.

        Paramters
        ---------
        image:
        iteration
        
        Returns
        -------
        image: np.ndarray
        loss: float
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
