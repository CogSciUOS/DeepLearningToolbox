"""Definition of the autoencoder interface and base class.

"""
# standard imports
from typing import Tuple, Optional

# third-party imports
import numpy as np

# toolbox imports
from dltb.base.implementation import Implementable
from dltb.base.run import runnable
from dltb.util.plot import plotting, TilingPlotter, Scatter2dPlotter


class Autoencoder(Implementable):
    """Base class for autoencoders.
    """

    # plyint: disable=too-few-public-methods,too-many-instance-attributes
    class Config:
        """Configuration values for the adversarial autoencoder.
        """
        model: str = 'semi_supervised'  # [supervised | semi_supervised]
        data: str = 'MNIST'  # [MNIST | CIFAR_10]
        prior: str = 'gaussian'  # [gaussain | gaussain_mixture | swiss_roll]

        super_n_hidden: int = 3000  # the number of elements for hidden layers
        semi_n_hidden: int = 3000  # the number of elements for hidden layers
        n_z: int = 20  # Dimension of Latent variables

        # number of samples for semi-supervised learning
        num_samples: int = 5000

        # training parameters
        n_epoch: int = 100  # number of Epoch for training
        batch_size: int = 128  # Batch Size for training

        keep_prob: float = 0.9  # dropout rate
        lr_start: float = 0.001  # initial learning rate
        lr_mid: float = 0.0001  # mid learning rate
        lr_end: float = 0.0001  # final learning rate

        noised: bool = True  # use noised data for training

        # additional program features
        flag_plot_mlr: bool = True  # plot manifold learning result
        flag_plot_arr: bool = False  # plot analogical reasoning result

    @property
    def conf(self) -> 'Autoencoder.Config':
        """The config object for this AAE.
        """
        return self._conf

    def __init__(self, shape: Optional[Tuple[int]] = None,
                 code_dim: Optional[int] = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self._conf = self.Config()
        if code_dim is not None:
            self._conf.n_z = code_dim

        # parameter
        self._data_shape = shape  # (height, width, channel)

    @property
    def data_shape(self) -> int:
        """Dimensionality of the the code space.
        """
        return self._data_shape

    @property
    def code_dim(self) -> int:
        """Dimensionality of the the code space.
        """
        return self._conf.n_z

    #
    # Coding API
    #

    @runnable
    def encode(self, data: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Encode data using the encoder part of the autoencoder.

        Arguments
        ---------
        data:
            The data to be encoded.

        Result
        ------
        code:
            The codes obtained from the data.
        """
        return self._encode(data, batch_size)

    def _encode(self, data: np.ndarray, batch_size: int = 128) -> np.ndarray:
        # pylint: disable=unused-argument, no-self-use
        ...

    @runnable
    def decode(self, code: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Decode given code values into the data space using the decoder
        part of the autoencoder.

        Arguments
        ---------
        code:
            The codes to be decoded.

        Result
        ------
        data:
            The reconstructed data.
        """
        return self._decode(code, batch_size)

    def _decode(self, code: np.ndarray, batch_size: int = 128) -> np.ndarray:
        # pylint: disable=unused-argument, no-self-use
        ...

    @runnable
    def recode(self, data: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """Reconstruct data values using the autoencoder, that is first
        encode the data and the decode it back into the data space.

        Arguments
        ---------
        data:
            The data to be recoded.

        Result
        ------
        recoded:
            The reconstructed data.
        """
        return self._recode(data, batch_size)

    def _recode(self, data: np.ndarray, batch_size: int = 128) -> np.ndarray:
        # pylint: disable=unused-argument, no-self-use
        ...

    #
    # Visualization/plotting
    #

    @plotting(Scatter2dPlotter)
    def plot_data_codes_2d(self, data: np.ndarray,
                           labels: Optional[np.ndarray] = None,
                           plotter: Optional[Scatter2dPlotter] = None,
                           **kwargs) -> None:
        """Create a scatter plot showing the distribution of codes in a
        2-dimensional latent space.  The code values are created
        by encoding the given data.

        Arguments
        ---------
        data:
            The data to be encoded to obtain the code values.
        label:
            Optional label data.  If not `None`, it should be an
            array of the same lenght as `data`. The code points
            in the scatter plot are shown in different colors,
            representing their respective label.
        **kwargs:
            Additional argument to be passed to the plotting function
            :py:func:`plot_2d_scatter`.
        """
        if self.code_dim != 2:
            raise ValueError("Scatter plot is only supported for 2D latent "
                             f"space (not for {self.code_dim} dimensions)")

        code = self.encode(data)
        if labels is not None and labels.ndim == 2:  # one-hot encoding
            labels = np.argmax(labels[:len(code)], axis=1)

        # plot the collected data
        plotter.plot_scatter_2d(code, labels=labels, **kwargs)

    @plotting(TilingPlotter)
    def plot_recoded_images(self, inputs: np.ndarray,
                            targets: Optional[np.ndarray] = None,
                            labels: Optional[np.ndarray] = None,
                            plotter: Optional[TilingPlotter] = None,
                            **kwargs) -> None:
        """Plot recoded images in a 2D grid.  Images are passed
        through the autoencoder and the resconstructed images are
        plotted.

        Arugments
        ---------
        inputs:
        targets:
        labels:
        plotter:
        """
        recoded = self.recode(inputs) if labels is None else \
            self.recode(inputs, labels=labels)
        # plot the collected data
        plotter.plot_tiling(recoded, rows=10, columns=10, **kwargs)

    @plotting(TilingPlotter)
    def plot_decoded_codespace_2d(self, labels=None,
                                  rows: int = 10, columns: int = 10,
                                  plotter: Optional[TilingPlotter] = None,
                                  **kwargs) -> None:
        """Compute and plot the manifold learning results.

        This method assumes a 2D latent space and an image data space.
        It samples a regular grid from the 2D code space and decodes
        these codes to obtain images, which will then be displayed in
        a grid to give an impression how of what codes in different
        regions of the codes space represent.

        """
        if self.code_dim != 2:
            raise ValueError("Plott the manifold learning results is only "
                             "supported for 2D latent space "
                             "(not for {self.code_dim} dimensions)")

        # define a grid of points in the 2D latent space
        x_axis = np.linspace(-0.5, 0.5, columns)
        y_axis = np.linspace(0.5, -0.5, rows)
        codes = np.stack(np.meshgrid(x_axis, y_axis)).\
            transpose((1, 2, 0)).reshape((-1, 2))

        images = self.decode(codes) if labels is None else \
            self.decode(codes, labels=labels)

        # plot the results
        plotter.plot_tiling(images, rows, columns, **kwargs)
