"""Definition of some random distributions.

Demo 1:

from dltb.util.plot import Scatter2dPlotter
from dltb.util.distributions import gaussian, gaussian_mixture, swiss_roll

plotter = Scatter2dPlotter()

count = 10000
nclasses = 10

data, labels = gaussian(count, labels=nclasses)
plotter.plot_scatter_2d(data=data, labels=labels)

data, labels = gaussian_mixture(count, components=nclasses, labels=True)
plotter.plot_scatter_2d(data=data, labels=labels)

data, labels = swiss_roll(count, labels=nclasses)
plotter.plot_scatter_2d(data=data, labels=labels)

"""

# this code is borrowed from
# https://github.com/hwalsuklee/tensorflow-mnist-AAE/

# standard imports
from typing import Optional
from math import sin, cos, sqrt

# third-party imports
import numpy as np
# The numpy.random.Generator API seems to be new (numpy 1.22).
# For compatibility with older numpy installations, we resort to the old
# numpy.random functions ...
# from numpy.random import default_rng


def gaussian(count: int, dimensions: int = 2, labels: Optional[int] = None,
             mean: float = 0.0, std: float = 1.0,
             seed: int = None) -> np.ndarray:
    """Sample `count` points from a Gaussian distribution.
    """
    # rng = default_rng(seed=seed)
    np.random.seed(seed=seed)

    if labels is not None:
        if dimensions != 2:
            raise Exception("Dimensions must be 2 to label the distribution.")

        def sample(labels):
            # point_x, point_y = rng.normal(mean, std, (2,))
            point_x, point_y = np.random.normal(mean, std, (2,))
            angle = np.angle((point_x-mean) + 1j*(point_y-mean), deg=True)
            dist = np.sqrt((point_x-mean)**2+(point_y-mean)**2)

            # label 0
            if dist < 1.0:
                label = 0
            else:
                label = ((int)((labels-1)*angle)) // 360

                if label < 0:
                    label += labels-1

                label += 1

            return np.array([point_x, point_y]).reshape((2,)), label

        np_points = np.empty((count, dimensions), dtype=np.float32)
        np_labels = np.empty((count), dtype=np.int32)
        for idx in range(count):
            for dim in range(0, dimensions, 2):
                point, label = sample(labels)
                np_points[idx, dim:dim+2] = point
                np_labels[idx] = label
        return np_points, np_labels

    shape = (count, dimensions)
    # np_points = rng.normal(mean, std, shape).astype(np.float32)
    np_points = np.random.normal(mean, std, shape).astype(np.float32)
    return np_points


def gaussian_mixture(count, components: int, labels: bool = False,
                     x_var: float = 0.5, y_var: float = 0.1) -> np.ndarray:
    """Sample `count` points from a Gaussian mixture model.
    """
    dimensions = 2
    # if dimensions != 2:
    #     raise Exception("dimensions must be 2.")
    seed = None  # fresh, unpredictable entropy will be pulled from the OS
    # rng = default_rng(seed=seed)
    np.random.seed(seed=seed)

    def sample(point_x: float, point_y: float, label: int, components: int):
        shift = 1.4
        rad = 2.0 * np.pi * label / components

        # rotate point by 'rad'
        new_x = point_x * cos(rad) - point_y * sin(rad)
        new_y = point_x * sin(rad) + point_y * cos(rad)

        # shift point by 'shift'
        new_x += shift * cos(rad)
        new_y += shift * sin(rad)

        # return the new point
        return np.array([new_x, new_y]).reshape((2,))

    # sample point locations ...
    # points_x = rng.normal(0, x_var, (count, dimensions//2))
    # points_y = rng.normal(0, y_var, (count, dimensions//2))
    points_x = np.random.normal(0, x_var, (count, dimensions//2))
    points_y = np.random.normal(0, y_var, (count, dimensions//2))
    # ... and point labels
    # np_labels = rng.integers(components, size=count)
    np_labels = np.random.randint(components, size=count)

    # create the result by rotating all points
    np_points = np.empty((count, dimensions), dtype=np.float32)
    for idx in range(count):
        for dim in range(0, dimensions, 2):
            np_points[idx, dim:dim+2] = \
                sample(points_x[idx, dim//2], points_y[idx, dim//2],
                       np_labels[idx], components)

    return (np_points, np_labels) if labels else np_points


def swiss_roll(count, labels: Optional[int] = None,
               seed: Optional[int] = None) -> np.ndarray:
    """Sample `count` points from a swiss roll distribution.
    """
    dimensions = 2
    # if dimensions != 2:
    #     raise Exception("dimensions must be 2.")
    # rng = default_rng(seed=seed)
    np.random.seed(seed=seed)

    def sample(label, labels):
        # uni = (rng.uniform(0.0, 1.0) + label) / labels
        uni = (np.random.uniform(0.0, 1.0) + label) / labels
        radius = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        point_x = radius * cos(rad)
        point_y = radius * sin(rad)
        return np.array([point_x, point_y]).reshape((2,))

    np_points = np.zeros((count, dimensions), dtype=np.float32)
    # np_labels = rng.integers(labels, size=count)
    np_labels = np.random.randint(labels, size=count)

    for idx in range(count):
        for dim in range(0, dimensions, 2):
            np_points[idx, dim:dim+2] = sample(np_labels[idx], labels)
    return (np_points, np_labels) if labels else np_points
