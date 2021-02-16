# standard imports
from typing import Callable
import logging


# third party imports
# import tensorflow as tf

# toolbox imports
from ...base import Preparable
from .v1 import tensorflow as tf

# logging
LOG = logging.getLogger(__name__)

class KerasTensorflowModel(Preparable):
    """A base class for Keras/TensorFlow models. An instance of this class
    represents a dedicated TensorFlow environment consisting of a
    TensorFlow Graph and a TensorFlow Session. The method
    :py:meth:`run_tensorflow` can be used to run a python function in that
    context.

    There are two motivation for establishing a specific session:

    (1) We have to make sure that a TensorFlow model is always used
    with the same tensorflow graph and the same tensorflow session
    in which it was initialized.  Hence we create these beforehand
    (during :py:meth:`prepare`) and store them for later use
    (in model initialization and prediction).

    (2) The aggressive default GPU memory allocation policy of
    TensorFlow: per default, TensorFlow simply grabs (almost) all
    of the available GPU memory. This can cause problems when
    other Tools would also require some GPU memory.

    Private Attributes
    ------------------
    _tf_graph: tf.Graph
    _tf_session: tf.Session

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tf_graph = None
        self._tf_session = None

    def _prepare(self):
        super()._prepare()
        self._prepare_tensorflow()

    def _unprepare(self) -> None:
        self._unprepare_tensorflow()
        super()._unprepare()

    def _prepared(self) -> bool:
        return self._prepared_tensorflow() and super()._prepared()

    def _prepare_tensorflow(self) -> None:
        """Initialize the Keras sesssion in which the model is to be executed.

        This method should be called before the first Keras Model is
        created.

        """
        # The TensorFlow GPU memory allocation policy can be
        # controlled by session parameters. There exist different
        # options, for example:
        #   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        #   config = tf.ConfigProto(gpu_options=gpu_options)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            self._tf_session = tf.Session(config=config)

        # One may try to set this session as Keras' backend session,
        # but this may be problematic if there is more than one Keras
        # model, each using its own Session. Hence it seems more
        # reliable to explicityl set default graph and default session,
        # as it is done by the method `keras_run()`.
        # backend.set_session(self._tf_session)

    def _unprepare_tensorflow(self) -> None:
        if self._tf_graph is not None:
            with self._tf_graph.as_default():
                if self._tf_session is not None:
                    self._tf_session.close()
                # AssertionError: Do not use tf.reset_default_graph()
                # to clear nested graphs. If you need a cleared graph,
                # exit the nesting and create a new graph.
                #
                # This error messsage is displayed when you call
                # tf.reset_default_graph() in one of the following
                # scenarios:
                #  * Inside a with graph.as_default(): block.
                #  * Inside a with tf.Session(): block.
                #  * Between creating a tf.InteractiveSession and
                #    calling sess.close().
                #
                # Each of these scenarios involves registering a
                # default (and potentially "nested") tf.Graph object,
                # which will be unregistered when you exit the block
                # (or close the tf.InteractiveSession). Resetting the
                # default graph in those scenarios would leave the
                # system in an inconsistent state, so you should
                # ensure to exit the block (or close the
                # tf.InteractiveSession) before calling
                # tf.reset_default_graph().
                #
                # tf.reset_default_graph()
            self._tf_graph = None
        self._tf_session = None
        
    def _prepared_tensorflow(self) -> bool:
        return (self._tf_graph and self._tf_session) is not None       

    def run_tensorflow(self, function: Callable, *args, **kwargs):
        """Run a python function in the context of the TensorFlow graph and
        session represented by this :py:class:`KerasTensorflowModel`.

        """
        with self._tf_graph.as_default():
            with self._tf_session.as_default():
                result = function(*args, **kwargs)

        return result
