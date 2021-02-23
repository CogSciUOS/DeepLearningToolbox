"""Tools for creating and definding againts adversarial examples.

"""


# third party imports
import numpy as np

# Toolbox imports
from ..datasource import Datasource
from .tool import Tool
from .classifier import Classifier, ClassIdentifier


class Attacker(Tool):
    # pylint: disable=too-many-ancestors
    """An attacker can create adversarial attacks ("adversarial examples")
    for a given victim. Currently we assume that the victim is a
    classifier.

    """

    def attack(self, victim: Classifier, example: np.ndarray,
               correct: ClassIdentifier, target: ClassIdentifier = None,
               **kwargs) -> np.ndarray:
        """
        Perform an attack against a victim.

        Arguments
        ---------
        victim: Classifier
            The victim (classifier) to attack.
        example: np.ndarray
            The example that should be altered.
        correct: ClassIdentifier
            Correct class label for the example.
        target: ClassIdentifier
            Target label for the attack. If none is given,
            an untargeted attack is done.

        Result
        ------
        adversarial_example: np.ndarray
            The adversarial example created by the attack.
        """


class Defender(Tool):
    # pylint: disable=too-many-ancestors
    """A :py:class:`Defender` aims at making a victim more robust
    against adversarial attacks.
    """

    def defend(self, victim: Classifier, attacker: Attacker = None,
               datasource: Datasource = None, **kwargs) -> None:
        """Defend the victim against adversarial attacks.
        """
