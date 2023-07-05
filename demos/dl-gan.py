#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for running and testing classifiers.

.. moduleauthor:: Ulf Krumnack


   dl-gan --model=celebahq --seed=6713


"""

# standard imports
import os
import sys
import random
import argparse
from threading import Thread

# third party imports
import numpy as np

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.util.image import ImageDisplay, imshow, imwrite
from dltb.tool.generator import ImageGAN



class Slideshow:
    """Present multiple images in a Slideshow
    """

    def __init__(self, gan) -> None:
        self._gan = gan
        self._shape = gan.data_shape
        self._canvas = np.zeros(self._shape, dtype=np.uint8)
        self._batches = []
        self._offset = 0
        while len(self) < 3:
            self._refill()

    def add_batch(self, batch) -> None:
        self._batches.append(batch)

    def step(self, step: int = 10):
        self._offset += step
        width = self._shape[1]
        height = self._shape[0]
        if self._offset >= width:
            del self._batches[0]
            self._offset -= width
            self.refill()
        self._canvas[:, 0:width-self._offset] = \
            (self._batches[0][0, :, self._offset:] if len(self._batches) > 0
             else np.zeros((height, width - self._offset, 3)))
        self._canvas[:, width-self._offset:] = \
            (self._batches[1][0, :, :self._offset] if len(self._batches) > 1
             else np.zeros((height, self._offset, 3)))

    @property
    def canvas(self) -> np.ndarray:
        return self._canvas

    def __len__(self) -> int:
        return len(self._batches)

    def refill(self) -> None:
        if len(self) < 3:
            Thread(name="refill", target=self._refill).start()

    def _refill(self) -> None:
        seed = random.randint(0, 10000)
        batch = self._gan.generate_array(seed=[seed])
        print("new batch:", batch.shape)
        self.add_batch(batch)


def main():
    """The main program.
    """
    parser = \
        argparse.ArgumentParser(description='GAN demonstration program')
    parser.add_argument('--list', action='store_true',
                        default=False, help='list GAN implementations')
    parser.add_argument('--class', type=str, default='ImageGAN',
                        help='GAN implementation')

    parser.add_argument('--model', type=str, default=None,
                        help='use pretrained model')
    parser.add_argument('--cats', dest='model',
                        action='store_const', const='cats',
                        help='use the cats model')
    parser.add_argument('--cars', dest='model',
                        action='store_const', const='cars',
                        help='use the cars model')
    parser.add_argument('--bedrooms', dest='model',
                        action='store_const', const='bedrooms',
                        help='use the bedrooms model')
    parser.add_argument('--celebahq', dest='model',
                        action='store_const', const='celebahq',
                        help='use the celebahq model')
    parser.add_argument('--ffhq', dest='model',
                        action='store_const', const='ffhq',
                        help='use the ffhq model')

    parser.add_argument('--filename', type=str, default=None,
                        help='filename for loading model')

    parser.add_argument('--seed', type=int, default=None,
                        help='generate image for given seed')
    parser.add_argument('--random', action='store_true', default=False,
                        help='choose random seed for image generation')
    parser.add_argument('--transition', action='store_true', default=False,
                        help='transition between two images')

    parser.add_argument('--mix', type=int, default=None,
                        help='mix style of two images')

    parser.add_argument('--show', action='store_true', default=False,
                        help='show generated image(s)')
    parser.add_argument('--store', action='store_true', default=False,
                        help='store generated image to disk')

    ToolboxArgparse.add_arguments(parser)
    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    if args.list:
        print(ImageGAN.implementation_info())
        return os.EX_OK

    # Instantiation and initialization
    gan = ImageGAN(implementation=getattr(args, 'class'),
                   model=args.model, filename=args.filename)
    gan.info()

    if args.seed is not None:
        image = gan.generate(seed=args.seed)
        if args.show:
            imshow(image)
        if args.store:
            filename = gan.model + '-' + str(args.seed) + '.jpg'
            print(f"Writing image to '{filename}'")
            imwrite(filename, image)

    elif args.transition:
        transition = gan.transition(400, 602, steps=200)
        print(transition.array.shape)

        with ImageDisplay() as display:
            for index, image in enumerate(transition):
                display.show(image, timeout=.1,
                             title=f"{index}/{len(transition)}")

    elif args.mix:
        with ImageDisplay() as display:
            mix_max = 18
            for mix in range(mix_max):
                image = gan.mix(mix=mix)
                display.show(image, timeout=1., title=f"{mix}/{mix_max}")

    elif False:  # FIXME[hack]: add option "--slideshow" once this is finished
        slideshow = Slideshow(gan)
        with ImageDisplay() as display:
            # FIXME[todo]: context manager should show the window
            while True:  # not display.closed:
                slideshow.step(step=2)
                display.show(slideshow.canvas, timeout=.01, unblock='freeze',
                             title=f"Slideshow ({slideshow._offset})")
                if display.closed:
                    break  # FIXME[todo]

    else:
        # Display randomly generated images
        with ImageDisplay() as display:
            # FIXME[todo]: context manager should show the window
            while True:  # not display.closed:
                seed = random.randint(0, 10000)
                image = gan.generate(seed=seed)
                display.show(image, timeout=0, unblock='freeze',
                             title=f"Seed={seed} -> {image.array.shape}")
                if display.closed:
                    break  # FIXME[todo]

    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main())
