"""A style transfer module in the spirit of FIXME


* TensorFlow Tutorial: Neural style transfer
  https://www.tensorflow.org/tutorials/generative/style_transfer
"""

# FIXME[todo]: why 3 classes?
#  - StyleContentModel(tf.keras.models.Model)
#  - TensorflowStyletransferEngine
#  - StyletransferTool(ImageTool)
#

# standard imports
from typing import Tuple, Iterator, Union, Dict, List
import logging

# third party imports
import numpy as np
import tensorflow as tf
from PIL import Image

# toolbox imports
from dltb.base.image import ImageTool
from dltb.util.image import imimport

# logging
LOG = logging.getLogger(__name__)



class StyleContentModel(tf.keras.models.Model):
    """A keras model that allows to extract content and style
    activation values for a given input.

    Attributes
    ----------

    _model: keras.Model
    _model_preprocessor:
        A function that is applied to to preprocess the input.
        The function expects RGB images in range 0-255 provided
        as tf.Tensor of datatype float (!).

    _activation_model: keras.Model
        a
    """

    def __init__(self, style_layers: List[str] = [],
                 content_layers: List[str] = []):
        super().__init__()
        self._model = None
        self._activation_model = None
        self.set_layers(style_layers=style_layers,
                        content_layers=content_layers)
        self.prepare()

    def set_layers(self, style_layers: List[str] = None,
                   content_layers: List[str] = None) -> None:
        if style_layers is not None:
            self._style_layers = style_layers
        if content_layers is not None:
            self._content_layers = content_layers
        if self.prepared:
            self._prepare_activation_model()

    def prepare(self) -> None:
        self._model = tf.keras.applications.VGG19(include_top=False,
                                                  weights='imagenet')
        self._model.trainable = False
        self._model_preprocessor = tf.keras.applications.vgg19.preprocess_input
        self._prepare_activation_model()

    @property
    def prepared(self) -> bool:
        return self._model is not None

    @property
    def num_style_layers(self) -> int:
        return len(self._style_layers)

    @property
    def num_content_layers(self) -> int:
        return len(self._content_layers)

    def _prepare_activation_model(self) -> None:

        outputs = []
        for name in self._style_layers:
            output = self._model.get_layer(name).output
            # outputs += [self.GramLayer()(output)]
            outputs += [tf.keras.layers.Lambda(self.gram_matrix)(output)]

        outputs += [self._model.get_layer(name).output
                    for name in self._content_layers]
        self._activation_model = tf.keras.Model([self._model.input], outputs)

    def call(self, inputs: tf.Tensor) -> Dict[str, Dict[str, tf.Tensor]]:
        """Compute the loss values for given input image(s).

        Remark: The :py:meth:`call` method is part of the Keras
        :py:class:`Model` API: it will perform forward propagation it
        is invoked by the :py:class:`__call__` method after some
        initial book keeping.

        Arguments
        ---------
        inputs: tf.Tensor
            A batch of input images. Expects RGB images of type
            float in range [0,1]

        Result
        ------
        losses: dict

        """
        print(f"Inputs: {inputs.shape} [{type(inputs)}]")
        inputs = inputs * 255.0
        preprocessed_input = self._model_preprocessor(inputs)
        outputs = self._activation_model(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        # 'block5_conv2' -> shape=(, 18, 32, 512)
        content_dict = {content_name: value for content_name, value
                        in zip(self._content_layers, content_outputs)}

        # 'block1_conv1': shape=(, 64, 64)
        # 'block2_conv1': shape=(, 128, 128)
        # 'block3_conv1': shape=(, 256, 256)
        # 'block4_conv1': shape=(, 512, 512)
        # 'block5_conv1': shape=(, 512, 512)
        style_dict = {style_name: value for style_name, value
                      in zip(self._style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

    @staticmethod
    def gram_matrix(inputs: tf.Tensor) -> tf.Tensor:
        result = tf.linalg.einsum('bijc,bijd->bcd', inputs, inputs)
        input_shape = tf.shape(inputs)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    class GramLayer(tf.keras.layers.Layer):
        def __init__(self) -> None:
            super().__init__()
            self._num_locations = None

        def build(self, input_shape: Tuple) -> None:
            # input_shape can be (None, None, None, 64)
            if input_shape[1] is not None and input_shape[2] is not None:
                self._num_locations = tf.cast(input_shape[1] * input_shape[2],
                                              tf.float32)

        def call(self, inputs: tf.Tensor) -> tf.Tensor:
            """Compute the Gram matrix. For real vectors with Euclidean
            product this is simply the matrix G = V'V.

            In this case we have a (batch, width, height, channels) input.
            For each batch item, we flatten the spatial dimenension,
            obtaining c vectors of dimension w*h. From these we obtain a
            c*c Gram matrix.

            """
            # G[b,c1,c2] = sum_i sum_j A[b,i,j,c1] * A[b,i,j,c2]
            result = tf.linalg.einsum('bijc,bijd->bcd', inputs, inputs)
            if self._num_locations is None:
                input_shape = tf.shape(inputs)
                self._num_locations = tf.cast(input_shape[1] * input_shape[2],
                                              tf.float32)
            return result / self._num_locations


class TensorflowStyletransferEngine:
    """
    _style:
    _style_targets:
    _style_weight:
    
    _content:
    _content_targets:
    _content_weight:

    _max_dim = 512

    _extractor: StyleContentModel
    _optimizer: Adam

    _image: tf.Tensor (tf.Variable)
        Internal representation of the current image. The optimization
        will be based on this representation and update it.
        This is basically the same as the `image` property, but
        `image` will be updated based on `_internal_image`, hence
        there may by a slight delay.

    _losses_total: List
    _losses_style: List
    _losses_content: List
    """

    def __init__(self, style_layers, content_layers,
                 style_weight: float = 1e-2,
                 content_weight: float = 1e4) -> None:

        # stateless API
        self._optimizer = \
            tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        # initialize the model
        self._extractor = StyleContentModel(style_layers, content_layers)

        self._max_dim = 512

        self._style = None
        self._style_targets = None
        self._style_weight = style_weight

        self._content = None
        self._content_targets = None
        self._content_weight = content_weight

        self._image = None

        # stateful API
        self._losses_total = []
        self._losses_style = []
        self._losses_content = []

    def reset(self, image: np.ndarray = None) -> None:
        if image is None:
            if self._content is not None:
                image_internal = tf.zeros_like(self._content, dtype=tf.float32)
            else:
                image_internal = tf.zeros((100, 100), dtype=tf.float32)
        else:
            image_internal = self.preprocess(image, max_dim=self._max_dim)
        self._image = tf.Variable(image_internal)
        self._losses_total = []
        self._losses_style = []
        self._losses_content = []

    @property
    def image(self) -> np.ndarray:
        """The style image.
        """
        return self.postprocess(self._image)

    @property
    def style(self) -> np.ndarray:
        """The style image.
        """
        return self.postprocess(self._style)

    @style.setter
    def style(self, style: np.ndarray) -> None:
        """Set the style image.
        """
        self._style = self.preprocess(style, max_dim=self._max_dim)
        self._style_targets = self._extractor(self._style)['style']

    @property
    def content(self) -> np.ndarray:
        """The content image.
        """
        return self.postprocess(self._content)

    @content.setter
    def content(self, content: np.ndarray) -> None:
        """Set the content image.
        """
        self._content = self.preprocess(content, max_dim=self._max_dim)
        self._content_targets = self._extractor(self._content)['content']

    @staticmethod
    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def style_content_loss(self, outputs):
        """Compute style-, content-, and total loss fort the given
        outputs.

        Arguments
        ---------

        Result
        ------
        loss:
        style_loss:
        content_loss:
        """
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = \
            tf.add_n([tf.reduce_mean((style_outputs[name] -
                                      self._style_targets[name])**2)
                      for name in style_outputs.keys()])
        style_loss *= self._style_weight / self._extractor.num_style_layers

        content_loss = \
            tf.add_n([tf.reduce_mean((content_outputs[name] -
                                      self._content_targets[name])**2)
                      for name in content_outputs.keys()])
        content_loss *= \
            self._content_weight / self._extractor.num_content_layers
        loss = style_loss + content_loss
        return loss, style_loss, content_loss

    def preprocess(self, image: np.ndarray, max_dim: int = None,
                   variable: bool = True) -> tf.Tensor:
        """Convert a given image into the internal representation.

        Arguments
        ---------
        image: np.ndarray
            An image in standard format (RGB, uint8, 0-255).
        max_dim: int
            The maximal dimension of the image in any axis. If the
            image is larger, it will be resized keeping the aspect
            ratio.
        variable: bool
            A flag indicating if the image should become a variable.
            This is required for optimization.

        Result
        ------
        image: tf.Tensor
            Image in internal representation (RGB, float32, 0.0-1.0).
        """

        image_tf = tf.convert_to_tensor(image)
        # This will change the type to float32 and the range to 0.0-1.0
        image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)

        # check if image is too large, resize
        if max_dim is not None:
            shape = tf.cast(tf.shape(image_tf)[:-1], tf.float32)

            # this requires eager execution
            scale = max_dim / max(shape)
            new_shape = tf.cast(shape * scale, tf.int32)

            image_tf = tf.image.resize(image_tf, new_shape)

        # add a batch axis
        image_tf = image_tf[tf.newaxis, :]
        return tf.Variable(image_tf) if variable else image_tf

    def postprocess(self, image_internal: tf.Tensor) -> np.ndarray:
        """Convert a given image into the standard representation.

        Arguments
        ---------
        image: tf.Tensor
            Image in internal representation (RGB, float32, 0.0-1.0).

        Result
        ------
        image: np.ndarray
            An image in standard format (RGB, uint8, 0-255).
        """
        # (1) remove batch axis
        # (2) scale from [0,1] to [0,256]
        # (3) convert to numpy (uint8)
        return np.array(image_internal[0] * 255, dtype=np.uint8)

    #
    # stateless API
    #

    def __call__(self, image: tf.Tensor, clip: bool = True) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply one step of optimization to the given image

        Arguments
        ---------
        image: tf.Tensor
            Image in internal (TensorFlow) representation. This
            image will be updated by performing one optimization
            step.

        Result
        ------
        loss: tf.Tensor (float)
        style_loss: tf.Tensor (float)
        total_loss: tf.Tensor (float)
        """
        with tf.GradientTape() as tape:
            outputs = self._extractor(image)
            loss = self.style_content_loss(outputs)
            grad = tape.gradient(loss, image)

        # apply one step of gradient descent
        self._optimizer.apply_gradients([(grad, image)])

        # make sure image values stey in valid range
        if clip:
            image.assign(self.clip_0_1(image))

        # return the loss
        return loss

    #
    # stateful API
    #

    def __next__(self) -> None:
        # loss, style_loss, and content_loss are
        # 'tensorflow.python.framework.ops.EagerTensor'
        loss, style_loss, content_loss = self(self._image)
        self._losses_total.append(float(loss))
        self._losses_style.append(float(style_loss))
        self._losses_content.append(float(content_loss))
        print(f"next(internal): loss={loss} "
              f"(style={style_loss}/content={content_loss})")


class StyletransferTool(ImageTool):
    """A style transfer tool.

    The style transfer tool generates a new image from two input
    images.  One image provides the style, while the other image
    provides the content.

    When run in loop mode, the tool can provide the following
    information for every step:
    - image (np.ndarray): the current state of the image
    - loss_total (float): the total loss value
      (weighted sum of style loss and content loss)
    - loss_style (float): the style loss
    - loss_content (float): the content loss
    - train_step (int): the content loss

    Properties
    ----------

    _losses_total: list
        A list holding the total loss values

    _losses_style: list
        A list holding the
    """

    def __init__(self, style_layers, content_layers,
                 style_weight: float = 1e-2,
                 content_weight: float = 1e4) -> None:
        super().__init__()
        self._engine = \
            TensorflowStyletransferEngine(style_layers,
                                          content_layers,
                                          style_weight=style_weight,
                                          content_weight=content_weight)

    #
    # style/content
    #

    @property
    def style(self) -> np.ndarray:
        """The style image.
        """
        return self._engine.style

    @style.setter
    def style(self, style: Union[np.ndarray, str]) -> None:
        """Set the style image.
        """
        self._engine.style = imimport(style)

    @property
    def content(self) -> np.ndarray:
        """The content image.
        """
        return self._engine.content

    @content.setter
    def content(self, content: Union[np.ndarray, str]) -> None:
        """Set the content image.

        Arguments
        ---------
        content:
            The content image (either numpy array or filename/URL).
        """
        self._engine.content = imimport(content)

    #
    # stateless API
    #

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply one optimization step
        """
        internal = self._engine.preprocess(image, variable=True)
        _ = self._engine(internal)  # we don't use the return value (loss)
        return self._engine.postprocess(internal)

    def optimization(self, image: Union[np.ndarray, str]) \
            -> Iterator[np.ndarray]:
        """Perform optimization starting with the given image.
        """
        internal = self._engine.preprocess(imimport(image), variable=True)
        while True:
            _ = self._engine(internal)  # we don't use the return value (loss)
            yield self._engine.postprocess(internal)

    #
    # stateful API
    #

    def reset(self, image: Union[str, np.ndarray] = 'content') -> None:
        """Reset this :py:class:`StyletransferTool`. The input will
        be reset to the specified value.

        Arguments
        ---------
        image:
            Either one of the special string 'zeros', 'random', 'content',
            or 'style', the name of an image file or a numpy array holding
            an image in standard format (RGB, uint8, 0-255).
        """
        if isinstance(image, str):
            if image == 'zeros':
                image = np.zeros(self.content.shape, dtype=np.uint8)
            elif image == 'random':
                image = np.random.randint(0, 256, self.content.shape,
                                          dtype=np.uint8)
            elif image == 'content':
                image = self.content
            elif image == 'style':
                # FIXME[bug]: this does not work - check!
                # It seems that the iamge has to have the same size as
                # the content image - setting the style image raises
                # an execption in `style_content_loss`:
                # Incompatible shapes: [1,25,32,512] vs. [1,18,32,512] [Op:Sub]
                image == self.style
            else:
                image = imimport(image)
        self._engine.reset(image)
        self._image = self._engine.image

    def next(self) -> None:
        """Perform one optimization step.
        The result will be stored in the attribute `image` and observers
        will be informed that a new image is available.
        """
        next(self._engine)
        self._image = self._engine.image
        self._step += 1
        self.change('image_changed')


class Old:
    # FIXME[old]

    @staticmethod
    def tensor_to_image(tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)
