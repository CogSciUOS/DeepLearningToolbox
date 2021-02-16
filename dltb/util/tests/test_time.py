
# standard imports
from unittest import TestCase
from time import time, sleep
from math import isclose

# toolbox imports
from dltb.util.time import pacemaker


class ImageTest(TestCase):
    """Tests for the :py:mod:`dltb.util.image` module.
    """
    places = 2
    interval = 0.5
    steps = 3
    delay = 0.2

    def test_pacemaker(self) -> None:
        start_time = time()
        for i in pacemaker(range(self.steps), self.interval):
            current_time = time() - start_time
            self.assertAlmostEqual(i*self.interval, current_time, self.places)

    def test_pacemaker_absolute(self) -> None:
        start_time = time()
        for i in pacemaker(range(self.steps), self.interval, absolute=True):
            current_time = time() - start_time
            self.assertAlmostEqual(i*self.interval, current_time, self.places)
            sleep(self.delay)

    def test_pacemaker_relative(self) -> None:
        next_time = time()
        for i in pacemaker(range(self.steps), self.interval, absolute=False):
            self.assertAlmostEqual(time(), next_time, self.places)
            sleep(self.delay)
            next_time = time() + self.interval
