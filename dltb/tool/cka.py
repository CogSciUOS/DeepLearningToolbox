"""Abstract interface for Centered Kernel Alignment (CKA).  CKA is a
popular method to compare activation values of network layers.

"""

# standard imports
import pickle

# third party imports
import numpy as np

# toolbox imports
from .activation import ActivationComparison, Activationslike
from .. import config


class HilbertSchmidtIndependenceCriterion:
    """Implementation of the Hilbert-Schmidt Independence Criterion (HSCI).

    The HSCI was originally proposed in [1] as a statistic for determining
    whether two sets of variables are independent.

    [1] Arthur Gretton et al. (2005): ""Measuring Statistical Dependence
        with Hilbert-Schmidt Norms"
        https://doi.org/10.1007/11564089_7
    """

    @staticmethod
    def _center_similarity_matrix(matrix):
        """Center similarity matrix along rows and columns.
        """
        centered_matrix = matrix - matrix.mean(axis=0)
        centered_matrix -= centered_matrix.mean(axis=1)[:, np.newaxis]
        return centered_matrix

    def __call__(self, k_matrix, l_matrix, centered: bool = False,
                 sqroot: bool = False, biased: bool = False) -> float:
        """
        Arguments
        ---------
        k_matrix:
        l_matrix:
        centered:
            A flag indicating of the similarity matrices are already
            centered (i.e., obtained from centered data).  If `True`,
            no additional centering step will be done.  Setting this
            to `False` is harmless, as centering centered matrices again
            does not change them, but will only waste computing resources.
        """

        # make sure both matrices are compatible
        if k_matrix.shape != l_matrix.shape:
            raise ValueError("HSCI matrices have different shapes: "
                             f"{k_matrix.shape} vs. {l_matrix.shape}")
        if k_matrix.shape[0] != k_matrix.shape[1]:
            raise ValueError("HSCI matrices are non square matrices: "
                             f"shape={k_matrix.shape}")
        if not centered:
            k_matrix = self._center_similarity_matrix(k_matrix)
            l_matrix = self._center_similarity_matrix(l_matrix)
        hsic = self._hsic(k_matrix, l_matrix)
        return np.sqrt(hsic) if sqroot else hsic

    @staticmethod
    def _hsic(k_matrix, l_matrix) -> float:
        """To be implemented by subclasses.
        """
        return float(k_matrix == l_matrix) - 2  # dummy value


class CenteredKernelAlignment(ActivationComparison):
    """Abstract base class for Centered kernel alignment [1,2].

    [1] Corinna Cortes et al. (2012): "Algorithms for Learning Kernels
        Based on Centered Alignment"
    [2] Simon Kornblith et al. (2019): "Similarity of Neural Network
        Representations Revisited"
    [3] Thao Nguyen et al. (2021): "Do wide and deep networks learn the
        same things? Uncovering how neural network representations vary
        with width and depth"
    """

    hsic = HilbertSchmidtIndependenceCriterion()
    kernel1 = lambda x: x @ x.T
    kernel2 = lambda x: x @ x.T

    def __call__(self, *args, **kwargs) -> float:
        return self.cka(*args, **kwargs)

    def cka(self,
            activations1: Activationslike,
            activations2: Activationslike) -> float:
        # pylint: disable=unused-argument,no-self-use
        """Compute CKA for two activation vectors.

        Arguments
        ---------
        activations1:
            First sequence of N activation vectors.  The row activations1[n]
            contains the D1-dimensional activation vector for the n-th
            datapoint.  In total, there are N such activation vectors.
            When realized as an array, that array should have the shape
            (N, D1), not (D1, N).
        activations2:
            Second sequence activation vectors, that is an array of (N, D2).
            It is essential, that both activations arguments contain the same
            number of activation vectors, and the the order of activation
            vectors is the same, that is activations1[n] should contain
            the activation vector for the same input datapoint as
            activations1[n].

        Result
        ------
        cka:
            The empirical centered kernel alignment value computed for
            the given activation values.

        Raises
        ------
        ValueError:
            The computation failed. This may for example happen, if
            the two activation matrices have a different number of rows.
        """
        if len(activations1) != len(activations2):
            raise ValueError("Different number of datapoints for "
                             f"activations1 {activations1.shape} and "
                             f"activations2 {activations2.shape}")

        return self._cka(activations1, activations2)

    def _cka(self,
             activations1: Activationslike,
             activations2: Activationslike) -> float:
        k_matrix = self.kernel1(activations1)
        l_matrix = self.kernel2(activations2)
        return (self.hsic(k_matrix, l_matrix, sqroot=False) /
                (self.hsic(k_matrix, k_matrix, sqroot=True) *
                 self.hsic(l_matrix, l_matrix, sqroot=True)))

    def _cka_matrix(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Compute the CKA for two activation matrices.  Activation
        matrices typically occur in convolutional networks. This
        function flattens the matrices and the applies CKA to the
        resulting activation vectors.

        Arguments
        ---------
        matrix1:
            The first stack of activation maps.  In typical 2D convolutional
            neural network such an activation map has three axes
            (height, width, channels).
            If provided as an array, it should have the shape (N, H, W, C),
            that is, the first axis should iterate the different maps.
        matrix2:
            The second stack of activation maps.

        Result
        ------
            The empirical centered kernel alignment value computed for
            the given activation values.
        """
        # FIXME[todo]: instead of flattening the activation map,
        # that is (N, H, W, C) -> (N, H*W*C), it may be more appropriate
        # to actually interpret each activation map (H, W, C) as a
        # collection of H*W feature vectors of dimension C,
        # that is (N, H, W, C) -> (N*H*W, C).
        # This would account for the fact, that at different locations
        # the same kernel is applied to compute the activation value.

        # unfold the activations, that is make a (n, h*w*c) representation
        shape = matrix1.shape
        activations1 = \
            np.reshape(matrix1, newshape=(shape[0], np.prod(shape[1:])))

        shape = matrix2.shape
        activations2 = \
            np.reshape(matrix2, newshape=(shape[0], np.prod(shape[1:])))

        # calculate the CKA score
        return self.cka(activations1, activations2)

    @staticmethod
    def _demo_activations(number: int) -> Activationslike:
        """

        Load up second hidden layer of MNIST networks from SVCCA's saved
        activations and compare.
        Please download activations from [1]

        [1] https://github.com/google/svcca/
            tree/master/tutorials/model_activations/MNIST
        """
        svcca_root = config.github_directory / 'svcca'
        if not svcca_root.is_dir():
            raise FileNotFoundError(f"The SVCCA directory '{svcca_root}' does"
                                    " not exist.")
        mnist_data = svcca_root / 'tutorials' / 'model_activations' / 'MNIST'

        if number == 1:  # model_0 from the SVCCA MNIST demo
            with open(mnist_data / 'model_0_lay02.p', 'rb') as act_file:
                return pickle.load(act_file)
        if number == 2:  # model_1 from the SVCCA MNIST demo
            with open(mnist_data / 'model_1_lay02.p', 'rb') as act_file:
                return pickle.load(act_file)

        raise ValueError(f"Invalid activations number: {number}")

    def _activations_from_keras_model(self, models, data) -> Activationslike:
        """
        """

    def demo(self) -> None:
        """Simple demo to apply CKA on demo data.
        """
        acts1 = self._demo_activations(1)
        acts2 = self._demo_activations(2)
        cka_score = self.cka(acts1, acts2)
        print(cka_score)

    def demo1(self):
        """Run the code on demo activation values from the SVCCA repository.

        """
        acts1 = self._demo_activations(1)
        acts2 = self._demo_activations(2)
        print("activation shapes", acts1.shape, acts2.shape)

        # Transpose one of the matrices for matrix multiplication
        # because the above implementation expects one of them to be
        # transposed already
        acts1_tp = np.transpose(acts1, (1, 0))
        acts2_tp = np.transpose(acts2, (1, 0))
        print("Activation shape:", acts1_tp.shape, acts2_tp.shape)

        # calculate CKA similarity score
        cka_similarity = self.cka(acts1, acts2)  # _tp
        print("CKA similarity:", cka_similarity)

        # checking how cka() works post accommodating transposed matrix there
        # calculate CKA similarity score
        cka_similarity = self.cka(acts1, acts2)
        print("CKA similarity:", cka_similarity)

    def demo2(self) -> None:
        """
        Krupal:
        * Call calculate_CKA_for_two_matrices() when you have two sets of
          activations.
        * Call compare_activations() and compare_activations2() and also
          use "plot similarity ..." when you have the whole network -
          in other words, when you want to make a grid type plot comparing
          each layer with another.
        """

        acts1 = self._demo_activations(1)
        acts2 = self._demo_activations(2)

        # calculate CKA similarity score using second implementation
        # cka_similarity_w_second_implementation = cka(acts1, acts2_tp)
        cka_similarity_w_second_implementation = \
            self._cka_matrix(acts1, acts2)
        print(cka_similarity_w_second_implementation)

        # calculating CKA similarity score between the same set
        # expecting it to be high
        cka_similarity_w_second_implementation = \
            self._cka_matrix(acts1, acts1)
        print(cka_similarity_w_second_implementation)

    def demo3(self) -> None:
        """Demonstrate the effect of applying SVD as SVCCA before CKA - as
        there is low correlation using CKA:

        """
        # pylint: disable=invalid-name, too-many-locals

        acts1 = self._demo_activations(1)
        acts2 = self._demo_activations(2)

        # Mean subtract activations
        cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
        cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

        # Perform SVD
        _U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
        _U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

        svacts1 = np.dot(s1[:20]*np.eye(20), V1[:20])
        # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
        svacts2 = np.dot(s2[:20]*np.eye(20), V2[:20])
        # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)

        print(svacts1.shape)
        print(svacts2.shape)

        # Transpose for matrix multiplication
        svacts2_tp = np.transpose(svacts2, (1, 0))

        # First implementation:
        cka_similarity = self.cka(svacts1, svacts2_tp)
        print(cka_similarity)

        # Second implementation:
        cka_similarity_w_second_implementation = \
            self._cka_matrix(svacts1, svacts2)
        print(cka_similarity_w_second_implementation)

    @staticmethod
    def _plot_helper(arr, xlabel, ylabel):
        plt.plot(arr, lw=2.0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()


###############################################################################
# Representational_Similarity.ipynb
# https://github.com/Krupal09/ICLR_Representation_Similarity



###############################################################################
# Implementation 1
# GitHub: Krupal09/ICLR_Representation_Similarity
# File: Representational_Similarity.ipynb
# Function: cka(kmat_1, kmat_2)

"""Representational Similarity:

Comparing representational similarity of the layers can give insights
about their learning similarities. [Li et al., 2015] compared
different neural networks based on the correlation among layers. This
work shows that layers in the beginning and at the end of the networks
correlate well but intermediate layers have very low correlations
among them. Another notable work in this area is [Raghu et al., 2017]
where they introduced a method to compare networks using singular
value decomposition followed by the canonical correlation analysis
(CCA) [Harold, 1936], hence SVCCA. [Raghu et al., 2017] showed that
CNNs broadly converge bottom up contributing to an evidential basis to
freezing early layers in the transfer learning process and the same
for RNNs have been reported by [Morcos et al., 2018].

Projection Weighted CCA (PWCCA)[Morcos et al., 2018] enhanced SVCCA by
associating higher weights to more important dimensions in order to
derive important dimensions claiming to separate signal and noise.

CCA and its variants(as discussed above) require a large amount of
data which is usually difficult to get in most of the
applications. [Kornblith et al., 2019] introduced Centered Kernel
Alignment(CKA) to overcome this drawback by providing a means to
compare representations' similarities with less amount of data.

Network activations and parts of the script are adapted from SVCCA's
github repo available at https://github.com/google/svcca CKA
implementation adapted from https://github.com/amzn/xfer/
https://github.com/aky4wn/convolutions-for-music-audio can also be
referred for CKA implimentation.

"""

# pylint: disable=ungrouped-imports,wrong-import-position,wrong-import-order
# pylint: disable=reimported
from pathlib import Path
import sys
import os
import pickle

# third party imports
import numpy as np
from matplotlib import pyplot as plt


class HSIC1(HilbertSchmidtIndependenceCriterion):
    """Simple numpy-based implementation of a (biased) Hilbert Schmidt
    Independence Criterion (HSIC) estimator.
    """

    @staticmethod
    def _hsic(k_matrix: np.ndarray, l_matrix: np.ndarray) -> float:
        normalization = 1/(len(k_matrix)-1)**2
        hsic = normalization * np.trace(k_matrix @ l_matrix)
        return hsic


class CKA1(CenteredKernelAlignment):
    """CKA Implementation - 1:

    Based on the paper's [1] official code [2].

    [1] Similarity of Neural Networks with Gradients
        https://arxiv.org/pdf/2003.11498.pdf
    [2] https://github.com/amzn/xfer/
    """

    def __init__(self, hsic=None) -> None:
        super().__init__()
        self.hsic = hsic or HSIC1()

    @staticmethod
    def _centering(kmat: np.ndarray) -> np.ndarray:
        """
        Centering the kernel matrix
        """
        return (kmat - kmat.mean(axis=0, keepdims=True) -
                kmat.mean(axis=1, keepdims=True) + kmat.mean())

    def _cka(self,
             activations1: Activationslike,
             activations2: Activationslike) -> float:
        r"""
        Compute the Centered Kernel Alignment between two kernel matrices.
        \rho(K_1, K_2) = \Tr (K_1 @ K_2) / ||K_1||_F / ||K_2||_F
        """
        kmat_1 = self._centering(activations1 @ activations1.T)
        kmat_2 = self._centering(activations2 @ activations2.T)

        # Remark: np.linalg.norm computes the Frobenius norm, that is
        # ||M||_F = \sqrt{\sum_{ij} M_{ij}**2}
        # for symmetric M this is the same as \sqrt{tr(M@M)}
        # Remark: the normalizing constant 1/(n-1)**2 in nominator
        # and denominator cancel out
        return np.trace(kmat_1 @ kmat_2) \
            / np.linalg.norm(kmat_1) / np.linalg.norm(kmat_2)


###############################################################################
# Implementation 2
# GitHub: Krupal09/ICLR_Representation_Similarity
# File: Representational_Similarity.ipynb
# Function: CKA(X, Y):

# pylint: disable=wrong-import-position, reimported
import tensorflow as tf
import numpy as np


class HSIC2(HilbertSchmidtIndependenceCriterion):
    """Unbiased HSIC estimation according to equation (3) in [1].
    Implementation adapted from [2]. This implementation should
    allow for minibatch CKA computation, so that the CKA value
    is independent of the batch sizes.


    Remark: this implementation yields significant different results
    than the biased estimator implemented in :py:class:`HSIC1`.  It is
    not clear yet, if this is desired, or due to some error in the
    implementation.

    References
    ----------
    [1] Thao Nguyen et al. (2021): "Do wide and deep networks learn the
        same things? Uncovering how neural network representations vary
        with width and depth"
    [2] https://github.com/phrasenmaeher/cka
        Notebook: do_nns_learn_the_same%3F.ipynb

    """

    def _hsic(self, k_matrix: np.ndarray, l_matrix: np.ndarray) -> float:
        """Computes an unbiased estimator of HISC. This is equation (2) from
        the paper.

        Arguments
        ---------
        k_matrix:
            The NxN Gram matrix (K=XX^T) obtained from a NxD1 data
            (activation) matrix X.
        l_matrix:
            The NxN Gram matrix (L=YY^T) obtained from a NxD2 data
            (activation) matrix Y.
        """
        # pylint: disable=invalid-name

        # create the unit **vector** filled with ones
        n = k_matrix.shape[0]
        ones = np.ones(shape=(n,))

        # fill the diagonal entries with zeros
        np.fill_diagonal(k_matrix, val=0)  # this is now K_tilde
        np.fill_diagonal(l_matrix, val=0)  # this is now L_tilde

        # first part in the square brackets
        trace = np.trace(np.dot(k_matrix, l_matrix))

        # middle part in the square brackets
        nominator1 = np.dot(np.dot(ones.T, k_matrix), ones)
        nominator2 = np.dot(np.dot(ones.T, l_matrix), ones)
        denominator = (n-1) * (n-2)
        middle = np.dot(nominator1, nominator2) / denominator

        # third part in the square brackets
        multiplier1 = 2/(n-2)
        multiplier2 = np.dot(np.dot(ones.T, k_matrix), np.dot(l_matrix, ones))
        last = multiplier1 * multiplier2

        # complete equation
        unbiased_hsic = 1/(n*(n-3)) * (trace + middle - last)

        return unbiased_hsic


class CKA2(CenteredKernelAlignment):
    """
    CKA Implementation - 2 : Based on wide and deep neural networks

    Adapted from [1]

    [1] https://github.com/phrasenmaeher/cka
        Notebook: do_nns_learn_the_same%3F.ipynb
    """

    def __init__(self, hsic=None) -> None:
        super().__init__()
        self.hsic = hsic or HSIC2()

    def _cka(self,
             activations1: Activationslike,
             activations2: Activationslike) -> float:
        """Computes the CKA of two matrices. This is equation (1) from the
        paper.
        """
        nominator = self.hsic(np.dot(activations1, activations1.T),
                              np.dot(activations2, activations2.T))
        denominator1 = self.hsic(np.dot(activations1, activations1.T),
                                 np.dot(activations1, activations1.T))
        denominator2 = self.hsic(np.dot(activations2, activations2.T),
                                 np.dot(activations2, activations2.T))
        cka = nominator/np.sqrt(denominator1*denominator2)

        return cka

    @staticmethod
    def _get_all_layer_outputs_fn(model):
        """Builds and returns function that returns the output of every
        (intermediate) layer for a given input."""
        return tf.keras.backend.function([model.layers[0].input],
                                         [l.output for l in model.layers[1:]])

    def _compare_activations(self, model1, model2, data_batch, progress=None):
        """Calculate a pairwise comparison of hidden representations and
        return a matrix.

        """

        # get function to get the output of every intermediate layer,
        # for model1 and model2
        outputs1 = self._get_all_layer_outputs_fn(model1)(data_batch)
        outputs2 = self._get_all_layer_outputs_fn(model2)(data_batch)
        return self._compare_activations2(outputs1, outputs2,
                                          progress=progress)

    def _compare_activations2(self, outputs1, outputs2, progress=None):
        # create a placeholder array
        result_array = np.zeros(shape=(len(outputs1), len(outputs2)))
        if progress is None:
            progress = lambda x: x

        for idx1, output1 in enumerate(progress(outputs1)):
            for idx2, output2 in enumerate(progress(outputs2)):
                cka_score = \
                    self._cka_matrix(output1, output2)
                result_array[idx1, idx2] = cka_score
        return result_array

    def demo_models(self) -> None:
        """Compare two (Keras) models.
        """

        # FIXME[old]: currently not working (no data, wrong path, ...)
        # We need ResNet50 and ResNet101 models (trained on Cifar10).
        # These can obtained with code from dltb.thirdparty.tensorflow:
        #
        #    from dltb.thirdparty.tensorflow import create_resnet50
        #    from dltb.thirdparty.tensorflow import create_resnet101
        #    from dltb.thirdparty.tensorflow import train_model_on_cifar10
        #
        #    resnet50 = train_model_on_cifar10(create_resnet50)
        #    resnet101 = train_model_on_cifar10(create_resnet101)
        #
        # Trained models can be pickled using the following code:
        #
        #    from dltb.thirdparty.tensorflow import make_keras_picklable
        #    from dltb.thirdparty.tensorflow import save_model_as_pickle
        #
        #    make_keras_picklable()
        #    save_model_as_pickle(resnet50, 'resnet50.pkl')
        #    save_model_as_pickle(resnet101, 'resnet101.pkl')
        #
        # If pickled versions have already been stored, they can be loaded
        # using the following code:
        #
        #    from dltb.thirdparty.tensorflow save_model_as_pickle, unpack
        #
        #    resnet50 = load_pickled_model('resnet50.pkl')
        #    resnet101 = load_pickled_model('resnet101.pkl')

        model_path = Path('.')
        with open(model_path / 'resnet50.pkl', 'rb') as model_file:
            resnet50 = pickle.load(model_file)
        with open(model_path / 'resnet101.pkl', 'rb') as model_file:
            resnet101 = pickle.load(model_file)

        cifar10 = tf.keras.datasets.cifar10
        (x_train, _y_train), (x_test, _y_test) = cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0  # scale the data

        sim = self._compare_activations(resnet50, resnet101, x_train[:256])
        print("similarity of activations: ", sim)

        sim = self._compare_activations2(resnet50, resnet101)
        plt.figure(figsize=(30, 15), dpi=200)
        axes = plt.imshow(sim, cmap='magma', vmin=0.0, vmax=1.0)
        axes.axes.invert_yaxis()
        # plt.savefig(model_path / 'r50_r101.png', dpi=400)


###############################################################################
# Implementation 3
# GitHub: Krupal09/ICLR_Representation_Similarity
# File: Representational_Similarity.ipynb

# pylint: disable=wrong-import-position
import torch


class CKA3(CenteredKernelAlignment):
    """Torch based CKA implementation adapted from [1].

    [1] https://github.com/aky4wn/convolutions-for-music-audio/
        blob/main/Experiments/Similarity/NSynth-CKA.ipynb
    """
    def __init__(self, hsic=None) -> None:
        super().__init__()
        self.hsic = hsic or HSIC2()

    @staticmethod
    def _cka_torch(activations1: torch.Tensor,
                   activations2: torch.Tensor) -> torch.Tensor:
        num = torch.norm(activations2.T@activations1, p='fro')
        norm1 = torch.norm(activations1.T@activations1, p='fro')
        norm2 = torch.norm(activations2.T@activations2, p='fro')
        cka = ((num/norm1) * (num/norm2))
        return cka

    def _cka(self, activations1: Activationslike,
             activations2: Activationslike) -> float:
        # Convert to tensor
        # pylint: disable=no-member
        activations1 = torch.from_numpy(activations1).float()
        activations2 = torch.from_numpy(activations2).float()
        cka = self._cka_torch(activations1, activations2)
        return cka.numpy()


###############################################################################
# Implementation 4
# Code from Anatome project


# pylint: disable=wrong-import-position
import torch


class CKA4(CenteredKernelAlignment):
    """PyTorch implementation of Centered Kernel Analysis based on code
    from [1].  The code claims to be an implementation of CKA as
    described in [2].

    [1] https://github.com/moskomule/anatome/blob/master/anatome/similarity.py
    [2] Simon Kornblith et al. (2019): "Similarity of Neural Network
        Representations Revisited"
    """

    def __init__(self, hsic=None) -> None:
        super().__init__()
        self.hsic = hsic or HSIC1()

    @staticmethod
    def _zero_mean(data: torch.Tensor, dim: int) -> torch.Tensor:
        return data - data.mean(dim=dim, keepdim=True)

    @staticmethod
    def _debiased_dot_product_similarity(matrix: torch.Tensor,
                                         sum_row_x: torch.Tensor,
                                         sum_row_y: torch.Tensor,
                                         sq_norm_x: torch.Tensor,
                                         sq_norm_y: torch.Tensor,
                                         size: int) -> torch.Tensor:
        return (matrix
                - size / (size - 2) * (sum_row_x @ sum_row_y)
                + sq_norm_x * sq_norm_y / ((size - 1) * (size - 2)))

    @classmethod
    def linear_cka_distance(cls, activations1: torch.Tensor,
                            activations2: torch.Tensor,
                            reduce_bias: bool = False) -> torch.Tensor:
        """Compute linear CKA distance, that is one minus CKA value.x
        """
        return 1-cls.linear_cka(activations1, activations2,
                                reduce_bias=reduce_bias)

    @classmethod
    def linear_cka(cls, activations1: torch.Tensor,
                   activations2: torch.Tensor,
                   reduce_bias: bool = False) -> torch.Tensor:
        """Linear CKA (that is CKA with linear kernel) used in Kornblith et
        al. 2019.

        Arguments
        ---------
        activations1:
            input tensor of shape NxD1
        activations2:
            input tensor of shape NxD2
        reduce_bias:
            debias CKA estimator, which might be helpful when
            D is limited.

        Returns
        -------
        cka:
            The CKA value computed for teh tifen activations.
        """
        if activations1.size(0) != activations2.size(0):
            raise ValueError("activations1.size(0) == activations2.size(0) "
                             "is expected, but got "
                             f"activations1.size(0)={activations1.size(0)}, "
                             f"activations2.size(0)={activations2.size(0)} "
                             "instead.")

        activations1 = cls._zero_mean(activations1, dim=0)
        activations2 = cls._zero_mean(activations2, dim=0)
        dot_prod = (activations2.t() @ activations1).norm('fro').pow(2)
        norm_1 = (activations1.t() @ activations1).norm('fro')
        norm_2 = (activations2.t() @ activations2).norm('fro')

        if reduce_bias:
            size = activations1.size(0)
            # (activations1 @ activations1.t()).diag()
            sum_row_1 = torch.einsum('ij,ij->i', activations1, activations1)
            sum_row_2 = torch.einsum('ij,ij->i', activations2, activations2)
            sq_norm_1 = sum_row_1.sum()
            sq_norm_2 = sum_row_2.sum()
            dot_prod = \
                cls._debiased_dot_product_similarity(dot_prod,
                                                     sum_row_1, sum_row_2,
                                                     sq_norm_1, sq_norm_2,
                                                     size)
            norm_1 = \
                cls._debiased_dot_product_similarity(norm_1.pow_(2),
                                                     sum_row_1, sum_row_2,
                                                     sq_norm_1, sq_norm_2,
                                                     size)
            norm_2 = \
                cls._debiased_dot_product_similarity(norm_2.pow_(2),
                                                     sum_row_1, sum_row_2,
                                                     sq_norm_1, sq_norm_2,
                                                     size)
        return dot_prod / (norm_1 * norm_2)

    def _cka(self, activations1: Activationslike,
             activations2: Activationslike) -> float:
        # Convert to tensor
        # pylint: disable=no-member
        activations1 = torch.from_numpy(activations1).float()
        activations2 = torch.from_numpy(activations2).float()
        return float(self.linear_cka(activations1, activations2))


###############################################################################
# Code from cka.py (Krupal)

# pylint: disable=wrong-import-position
from datetime import datetime


def demo() -> None:
    """Compare layers in multi-path network for the ICLR paper.
    """
    # pylint: disable=invalid-name, too-many-locals

    cka = CKA1()

    scratch_base = Path('/net/projects/scratch')
    scratch_summer = scratch_base / 'summer' / 'valid_until_31_January_2022'
    # scratch_winter = scratch_base / 'winter' / 'valid_until_31_July_2022'

    # generate folder for all files using current timestamp
    folder = datetime.now().strftime("%d%b%Y-%H%M")
    folder = scratch_summer / 'kshah' / 'ICLR' / 'output' + folder
    os.mkdir(folder)

    # redirect all output and error to a file inside our folder
    sys.stdout = open(folder / 'output', 'w')
    sys.stderr = open(folder / 'error', 'w')

    # path to the latent dataset
    experiment = 'MPNet36_4_1_3_11d3_ImageNette_250'
    # experiment = 'MPNet18_4_1_3_7_Cifar10_64'
    latent_datasets = (scratch_summer / 'matrichter' / 'phd_lab' /
                       'phd_lab' / 'latent_datasets')
    path = latent_datasets / experiment
    x = [x for x in path.iterdir() if "stage" in str(x)]

    print()
    print("Path to the latent dataset being compared: ", path)
    print("Number of files to process with activations :", len(x))
    print()

    # visited_files_path: to avoid repeated comparisons like a to b and b to a
    visited_files_path = []

    for file_path in x:
        with open(file_path, "rb") as f:
            acts1 = pickle.load(f)

        file_names = str(file_path).split("/")   # directory/path/filename.p
        file_name = file_names[len(file_names)-1]  # filename.p

        # stage: train-stage1-0-pathway, pathway_layer: 0-layers-0-0.p
        stage, pathway_layer = file_name.split("pathway")
        pathway = pathway_layer[0]   # pathway: 0
        # layer = pathway_layer[2:len(pathway_layer)-2]  # layer: layers-0-0.p

        if file_name.startswith("train"):
            phase = "train"
        else:
            phase = "eval"

        # traverse through the entire list of pickle files and compare
        # only with the same phase - train or eval, same stage (and
        # substage) but different pathways
        for file_path_compare in x:

            if file_path_compare in visited_files_path:
                continue  # avoid repeated comparisons like a to b and b to a

            file_names_compare = str(file_path_compare).split("/")
            file_name_compare = file_names_compare[len(file_names_compare)-1]
            stage_compare, pathway_layer_compare = \
                file_name_compare.split("pathway")
            pathway_compare = pathway_layer_compare[0]   # pathway: 0
            # layer_compare = \
            #     pathway_layer_compare[2:len(pathway_layer_compare)-2]

            # code is not limited to compare only two pathways
            # no changes required while adding more pathways than two
            if file_name_compare.startswith(phase):
                if (stage == stage_compare) and (pathway != pathway_compare):
                    with open(file_path_compare, "rb") as f:
                        acts2 = pickle.load(f)

                    print("Comparison of ", file_name, "to ",
                          file_name_compare, ": ")

                    # calculate CKA similarity score
                    cka_similarity2 = cka.cka(acts1, acts2)
                    print("CKA similarity score :", cka_similarity2)
                    print()

        visited_files_path.append(file_path)


def test1():
    """Generate some data and check the function
    """
    samples = 100
    dim_x = 30
    dim_y = 40
    activations_x = np.random.randn(samples, dim_x)
    activations_y = np.random.randn(samples, dim_y)
    activations_x_permuted = activations_x[:, np.random.permutation(dim_x)]

    #
    # Step 1: test the HSIC implementations
    #
    tests = []
    tests.append(HSIC1())
    tests.append(HSIC2())

    k_matrix = activations_x @ activations_x.T
    l_matrix = activations_y @ activations_y.T

    print("Testing HSIC(K,K):")
    for idx, hsic in enumerate(tests):
        print(idx, f"{type(hsic).__name__}(K,K):",
              hsic(k_matrix, k_matrix))

    print("Testing HSIC(L,L):")
    for idx, hsic in enumerate(tests):
        print(idx, f"{type(hsic).__name__}(L,L):",
              hsic(l_matrix, l_matrix))

    print("Testing HSIC(K,L):")
    for idx, hsic in enumerate(tests):
        print(idx, f"{type(hsic).__name__}(K,L):",
              hsic(k_matrix, l_matrix))
    print()

    #
    # Step 2: test the CKA implementations
    #
    tests = []
    tests.append(CKA1())
    tests.append(CKA2())
    tests.append(CKA2(hsic=HSIC1()))
    tests.append(CKA3())
    tests.append(CKA4())

    print("Testing CKA(X,X):")
    for idx, cka in enumerate(tests):
        print(idx, f"{type(cka).__name__}(X,X):",
              cka(activations_x, activations_x))

    print("Testing CKA(X,permute(X)):")
    for idx, cka in enumerate(tests):
        print(idx, f"{type(cka).__name__}(X,permute(X)):",
              cka(activations_x, activations_x_permuted))

    print("Testing CKA(X,Y):")
    for idx, cka in enumerate(tests):
        print(idx, f"{type(cka).__name__}(X,Y):",
              cka(activations_x, activations_y))


if __name__ == '__main__':
    test1()
