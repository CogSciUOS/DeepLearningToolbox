"""Vanilla GAN model based on [1].

[1] https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
    https://github.com/diegoalejogm/gans/blob/master/1.%20Vanilla%20GAN%20PyTorch.ipynb
"""

# standard imports
from typing import Tuple
import os
import logging
from pathlib import Path

# third-party imports
import numpy as np
import torch
from torch import nn, optim
from torch.autograd.variable import Variable

# Toolbox imports
from ...tool.generator import ImageGAN
from ...base import Preparable
from ...base.busy import BusyObservable, busy

# Logging
LOG = logging.getLogger(__name__)

# FIXME[hack]
import os
from pathlib import Path

for directory in ('/net/store/cv/users/krumnack/var',
                  '/work/ulf/var'):
    directory = Path(directory)
    if directory.is_dir():
        data_dir = directory / 'data'
        break
else:
    data_dir = Path('.') / 'data'


dataset_dir = Path('.', 'dataset')

for directory in ('/space/data/mnist/raw',
                  '/net/projects/data/MNIST/original/raw'):
    directory = Path(directory)
    if directory.is_dir():
        dataset_mnist_raw_dir = directory
        break
else:
    dataset_mnist_raw_dir = Path('.', 'dataset', 'MNIST', 'raw')


class VGAN(ImageGAN, Preparable, BusyObservable):

    class DiscriminatorNet(torch.nn.Module):
        """A three hidden-layer discriminative neural network
        """
        def __init__(self, n_features: int = 784, n_out: int = 1,
                     **kwargs) -> None:
            super().__init__(**kwargs)

            self.hidden0 = nn.Sequential(
                nn.Linear(n_features, 1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            )
            self.hidden1 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            )
            self.hidden2 = nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            )
            self.out = nn.Sequential(
                torch.nn.Linear(256, n_out),
                torch.nn.Sigmoid()
            )

        def forward(self, x):
            x = self.hidden0(x)
            x = self.hidden1(x)
            x = self.hidden2(x)
            x = self.out(x)
            return x

    class GeneratorNet(torch.nn.Module):
        """
        A three hidden-layer generative neural network
        """
        def __init__(self, n_features: int = 100, n_out: int = 784,
                     **kwargs) -> None:
            super().__init__(**kwargs)

            self.hidden0 = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.LeakyReLU(0.2)
            )
            self.hidden1 = nn.Sequential(
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2)
            )
            self.hidden2 = nn.Sequential(
                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2)
            )

            self.out = nn.Sequential(
                nn.Linear(1024, n_out),
                nn.Tanh()
            )

        def forward(self, x):
            x = self.hidden0(x)
            x = self.hidden1(x)
            x = self.hidden2(x)
            x = self.out(x)
            return x

    models = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "99"]

    def __init__(self, model: str = "99", filename=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._generator = None
        self._discriminator = None

        self._feature_dimensions = 100
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, model: str) -> None:
        if self._model == model:
            return  # nothing todo

        self.unprepare()
        self._model = model

    def _data_shape(self) -> Tuple[int]:
        return (786, )

    def _prepared(self) -> bool:
        return self._discriminator is not None and super()._prepared()

    def _prepare(self) -> None:
        super()._prepare()

        discriminator = VGAN.DiscriminatorNet()
        generator = VGAN.GeneratorNet()

        if torch.cuda.is_available():
            discriminator.cuda()
            generator.cuda()

        #
        # load pretrained model
        #
        epoch = int(self._model)
        out_dir = data_dir / 'models/VGAN/MNIST'

        LOG.info("Loading pretrained model from '%s', epoch=%d",
                 out_dir, epoch)

        # FIXME[bug]:
        #  works fine with conda pytorch 1.7.0 (with build
        #  py3.7_cuda10.1.243_cudnn7.6.3_0) from the pytorch channel
        #  but fails with pytorch 1.4.0 (build cuda101py37h02f0884_0)
        #  from the default channel, with the RuntimeError:
        #    version_ <= kMaxSupportedFileFormatVersion INTERNAL ASSERT FAILED
        #      Attempted to read a PyTorch file with version 3,
        #      but the maximum supported version for reading is 2.
        #      Your PyTorch installation may be too old.
        generator.load_state_dict(torch.load(f'{out_dir}/G_epoch_{epoch}'))
        discriminator.load_state_dict(torch.load(f'{out_dir}/D_epoch_{epoch}'))

        self._generator = generator
        self._discriminator = discriminator

    def _unprepare(self) -> None:
        self._discriminator = None
        self._generator = None
        super()._unprepare()

    #
    # Sampling
    #

    def _images_to_vectors(self, images):
        return images.view(images.size(0), 784)

    def _vectors_to_images(self, vectors):
        return vectors.view(vectors.size(0), 28, 28)

    def noise(self, size):
        """Generate a 1-d vector of gaussian sampled random values.
        """
        n = Variable(torch.randn(size, 100))
        if torch.cuda.is_available():
            return n.cuda()
        return n

    def _randn(self, shape, seed=0):
        n = Variable(torch.randn(*shape))
        if torch.cuda.is_available():
            n = n.cuda()
        return n.data.cpu().detach().numpy()

    def generate_images(self):
        num_test_samples = 16
        test_noise = self.noise(num_test_samples)
        generator_output = self._generator(test_noise)
        discriminator_output = self._discriminator(generator_output)
        test_images = self._vectors_to_images(generator_output).data.cpu()
        print(test_images.shape, discriminator_output.shape)

    def _generate_batch(self, features: np.ndarray):
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if torch.cuda.is_available():
            features = features.cuda()
        generator_output = self._generator(features)
        images = self._vectors_to_images(generator_output).data.cpu()
        images = ((images + 1) * 255).clamp(0, 255).type(torch.uint8)
        print(images.shape, images.dtype, images.min(), images.max())
        return images.detach().numpy()

    #
    # Training
    #

    def _ones_target(self, size) -> None:
        """Tensor containing ones, with shape = size
        """
        data = Variable(torch.ones(size, 1))
        if torch.cuda.is_available():
            return data.cuda()
        return data

    def _zeros_target(size):
        """Tensor containing zeros, with shape = size
        """
        data = Variable(torch.zeros(size, 1))
        if torch.cuda.is_available():
            return data.cuda()
        return data

    def prepare_train(self) -> None:
        self._d_optimizer = \
            optim.Adam(self._discriminator.parameters(), lr=0.0002)
        self._g_optimizer = \
            optim.Adam(self._generator.parameters(), lr=0.0002)

        self._loss = nn.BCELoss()

    def train_discriminator(self, optimizer, real_data, fake_data):
        N = real_data.size(0)
        # Reset gradients
        optimizer.zero_grad()

        # 1.1 Train on Real Data
        prediction_real = self._discriminator(real_data)
        # Calculate error and backpropagate
        error_real = self._loss(prediction_real, ones_target(N))
        error_real.backward()

        # 1.2 Train on Fake Data
        prediction_fake = self._discriminator(fake_data)
        # Calculate error and backpropagate
        error_fake = self._loss(prediction_fake, self._zeros_target(N))
        error_fake.backward()

        # 1.3 Update weights with gradients
        optimizer.step()

        # Return error and predictions for real and fake inputs
        return error_real + error_fake, prediction_real, prediction_fake

    def train_generator(self, optimizer, fake_data):
        N = fake_data.size(0)
        # Reset gradients
        optimizer.zero_grad()
        # Sample noise and generate fake data
        prediction = self._discriminator(fake_data)
        # Calculate error and backpropagate
        error = self._loss(prediction, self._ones_target(N))
        error.backward()
        # Update weights with gradients
        optimizer.step()
        # Return error
        return error

    def info(self) -> None:
        print(self._generator)
        print(self._discriminator)
