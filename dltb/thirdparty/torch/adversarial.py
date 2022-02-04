"""Torch code for creating adversarial examples.

Heavily inspired from a tutorial by savan77 [1].

Code
----

from dltb.thirdparty.torch.vision import Network
from dltb.thirdparty.torch.adversarial import IterativeGradientSignAttacker

victim = Network('inception_v3')
attacker = IterativeGradientSignAttacker()

image = 'examples/cat.jpg'
target = 123

adversarial, logits = attacker.attack(victim, image, target=target)



References
----------
[1] https://savan77.github.io/imagenet_adv_examples
"""


# standard imports
from typing import Tuple, Optional
from collections import namedtuple
import logging

# thirdparty imports
import numpy as np
import torch
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F

# toolbox imports
from dltb.tool import Context
from dltb.tool.classifier import Classifier, ClassIdentifier
from dltb.tool.adversarial import Attacker
from .network import Network as TorchNetwork

# logging
LOG = logging.getLogger(__name__)

Example = namedtuple('Example', 'inputs label score')


class TorchAttacker(Attacker):
    # pylint: disable=abstract-method
    """Base class for adversarial attacks against torch networks.

    Arguments
    ---------
    epsilon:
        The maximal absolute value by which a pixel is allowed to change.
    """
    # Context fields:
    # 'victim': The victim of type Classifier
    # '_victim': Internal representation of the victim (torch.nn.Module)
    #
    # 'original': Example structure describing the original input
    # 'original_float': Original image as float array
    # 'original_uint8': Original image as uint8 array
    # 'original_logits': Logits for the original image as float array
    # 'original_scores': Scores for the original image as float array
    # 'original_predicted': The class predicted for the original example
    # 'original_confidence': The confidence score for that prediction
    # 'original_': original example as torch.Tensor
    # 'original_logits_': Logits for the original image as torch.Tensor
    #
    # 'adversarial': Example structure describing the adversarial example
    # 'adversarial_float': Adversarial image as float array
    # 'adversarial_uint8': Adversarial image as uint8 array
    # 'adversarial_logits': Logits for the adversarial example as float array
    # 'adversarial_scores': Scores for the adversarial example as float array
    # 'adversarial_': adversarial example as torch.Tensor
    # 'adversarial_logits_': Logits for the adversarial example as torch.Tensor
    #
    # 'correct': class identifier for the original class
    # 'target': class identifier for the target class

    external_result: Tuple[str] = ('adversarial', )

    def __init__(self, epsilon=0.25, **kwargs) -> None:
        """
        """
        super().__init__(**kwargs)
        self._epsilon = epsilon

    def _preprocess(self, victim: Classifier, example: np.ndarray,
                    correct: ClassIdentifier = None,
                    target: ClassIdentifier = None,
                    **kwargs) -> Context:
        # FIXME[refactor]: we need to come up with a better processing scheme
        # to subclass the Tool class in a type save way.
        # pylint: disable=arguments-differ,too-many-arguments,too-many-locals
        context = super()._preprocess(**kwargs)

        if not isinstance(victim, TorchNetwork):
            raise TypeError("Currently only attacks on torch models"
                            f"({TorchNetwork.__module__}."
                            f"{TorchNetwork.__name__}) not on"
                            f"{type(victim).__module__}."
                            f"{type(victim).__name__} are implemented. Sorry!")
        context.add_attribute('victim', victim)
        context.add_attribute('_victim', victim.model)

        image_tensor = example if isinstance(example, torch.Tensor) else \
            victim.image_to_internal(example)
        image_tensor = image_tensor.to(victim.torch_device)
        context.add_attribute('original_', image_tensor)

        predicted, confidence = victim.classify(image_tensor, confidence=True)
        context.add_attribute('original_predicted', predicted)
        context.add_attribute('original_confidence', confidence)

        if correct is None and target is None:
            correct = predicted

        context.add_attribute('correct', None if correct is None else
                              victim.class_identifier(correct))
        context.add_attribute('_correct', correct)
        context.add_attribute('target', None if target is None else
                              victim.class_identifier(target))
        context.add_attribute('_target', target)

        context.add_attribute('_lower_bound',
                              victim.lower_bound.view(1, 3, 1, 1).
                              to(victim.torch_device))
        context.add_attribute('_upper_bound',
                              victim.upper_bound.view(1, 3, 1, 1).
                              to(victim.torch_device))

        return context

    def _postprocess(self, context: Context, what: str) -> None:
        # pylint: disable=too-many-branches
        victim = context.victim
        if what == 'original_logits':
            logits = context.original_logits_.cpu().numpy()
            context.add_attribute(what, logits)
        elif what == 'original_float':
            original_ = context.original_
            context.add_attribute(what, victim.internal_to_image(original_))
        elif what == 'original_predicted':
            predicted, confidence = \
                victim.classify(context.original_, confidence=True)
            context.add_attribute('original_predicted', predicted)
            context.add_attribute('original_confidence', confidence)
        elif what == 'original':
            for requirement in ('original_float', 'original_predicted'):
                if not hasattr(context, requirement):
                    self._postprocess(context, requirement)
            predicted = context.original_predicted
            example = Example(context.original_float, f"{predicted} "
                              f"({victim.get_label(predicted, 'text')})",
                              context.original_confidence)
            context.add_attribute(what, example)
        elif what == 'adversarial_logits':
            logits = context.adversarial_logits_.cpu().numpy()
            context.add_attribute(what, logits)
        elif what == 'adversarial_float':
            adversarial_ = context.adversarial_
            context.add_attribute(what, victim.internal_to_image(adversarial_))
        elif what == 'adversarial_predicted':
            predicted, confidence = \
                victim.classify(context.adversarial_, confidence=True)
            context.add_attribute('adversarial_predicted', predicted)
            context.add_attribute('adversarial_confidence', confidence)
        elif what == 'adversarial':
            for requirement in ('adversarial_float', 'adversarial_predicted'):
                if not hasattr(context, requirement):
                    self._postprocess(context, requirement)
            predicted = context.adversarial_predicted
            example = Example(context.adversarial_float, f"{predicted} "
                              f"({victim.get_label(predicted, 'text')})",
                              context.adversarial_confidence)
            context.add_attribute(what, example)
        else:
            super()._postprocess(context, what)

    def attack(self, victim: Classifier, example: np.ndarray,
               correct: ClassIdentifier = None, target: ClassIdentifier = None,
               **kwargs) -> np.ndarray:
        """
        Perform an attack against a victim.

        Arguments
        ---------
        victim: Classifier
            The victim (torch classifier) to attack.
        example: np.ndarray
            The example that should be altered.
        correct: ClassIdentifier
            Correct class label for the example. Can be given in
            case of untargeted attacks.  The adversarial example
            will then have a small confidence value for that class.
            If `None`, the model prediction is assumed to be the
            correct class.
        target: ClassIdentifier
            Target label for the attack. If `None`,
            an untargeted attack is done.

        Result
        ------
        adversarial_example: np.ndarray
            The adversarial example created by the attack.
        """
        return self(victim, example, correct=correct, target=target, **kwargs)


class FastGradientSignAttacker(TorchAttacker):
    """Implementation of the fast (i.e., one step) gradient sign method.
    """

    internal_arguments: Tuple[str] = ('_victim', 'original_',
                                      '_correct', '_target',
                                      '_lower_bound', '_upper_bound')
    internal_result: Tuple[str] = ('adversarial_', 'adversarial_logits_',
                                   'original_logits_')

    def _process(self, victim: torch.nn.Module, original: torch.Tensor,
                 correct: ClassIdentifier,
                 target: ClassIdentifier,
                 vmin: Optional[torch.Tensor],
                 vmax: Optional[torch.Tensor]) -> Tuple[torch.Tensor,
                                                        torch.Tensor,
                                                        torch.Tensor, int]:
        # FIXME[refactor]: we need to restructure this class and come up with
        # a better processing scheme to subclass the Tool class ...
        # pylint: disable=arguments-differ,too-many-arguments,too-many-locals
        """
        """
        ascent = False

        # Create a tensor to be adapted to obtain to hold the
        # adversarial example.
        example = original.clone()
        example.requires_grad_(True)
        zero_gradients(example)

        # obtain initial logits
        original_logits = victim.forward(example)

        # if no target is given, choose the class with lowest
        # confidence score (can be considered the hardest task)
        if target is None:
            if correct is None:
                target = torch.min(original_logits, 1)[1][0]
            else:
                target = correct
                ascent = True

        # create a tensor for the target class
        # pylint: disable=not-callable
        # torch bug: https://github.com/pytorch/pytorch/issues/24807
        y_target = \
            torch.tensor([target], dtype=torch.long, device=example.device)

        # the loss function
        loss = torch.nn.CrossEntropyLoss()
        loss_cal = loss(original_logits, y_target)

        # this will calculate gradient of each variable
        # (with requires_grad=True) and can be accessed by "var.grad.data"
        loss_cal.backward(retain_graph=True)

        # update the example
        with torch.no_grad():
            # calculate the sign of gradient of the loss func
            # (with respect to original) and multply by stepwidth
            update = self._alpha * torch.sign(example.grad)

            if ascent:
                # do gradient ascent (increase the loss for the target class)
                example += update
            else:
                # do gradient descent (reduce the loss for the target class)
                example -= update

            # make sure that the resulting image is not too different from
            # the original input data (image)
            if self._epsilon is not None:
                total_diff = torch.clamp(example - original,
                                         -self._epsilon, self._epsilon)
                example.copy_(original + total_diff)

            # Clamping the values to stay in valid range
            if vmin is not None:
                example.copy_(torch.max(example, vmin))
            if vmax is not None:
                example.copy_(torch.min(example, vmax))

        # obtain new logits
        with torch.no_grad():
            logits = victim.forward(example)

        return example.detach(), logits.detach(), original_logits.detach()


class IterativeGradientSignAttacker(TorchAttacker):
    """Implementation of the iterative gradient sign method.
    This implementation is base on [1].

    Arguments
    ---------
    alpha:
        The step size by which the input is changed in each iteration.
    max_steps:
        The maximal number of steps to perform. May stop earlier if
        an adversarial example was created before.
    min_confidence:
        The minimal confidence for the target class.


    References
    ----------
    [1] https://savan77.github.io/imagenet_adv_examples
    """

    internal_arguments: Tuple[str] = ('_victim', 'original_',
                                      '_correct', '_target',
                                      '_lower_bound', '_upper_bound')
    internal_result: Tuple[str] = ('adversarial_', 'adversarial_logits_',
                                   'original_logits_', 'steps')

    iterative_arguments: Tuple[str] = ('adversarial_', 'adversarial_logits_',
                                       'steps')

    # Context fields (in addition to the fields from TorchAttacker):
    # 'steps': number of steps the algorithm has run'

    def __init__(self, alpha: float = 0.025, max_steps: int = 5,
                 min_confidence: Optional[float] = None, **kwargs) -> None:
        """
        """
        super().__init__(**kwargs)
        self._alpha = alpha
        self._max_steps = max_steps
        self._min_confidence = min_confidence

    def _process(self, victim: torch.nn.Module, original: torch.Tensor,
                 correct: int, target: int,
                 vmin: Optional[torch.Tensor],
                 vmax: Optional[torch.Tensor]) -> Tuple[torch.Tensor,
                                                        torch.Tensor,
                                                        torch.Tensor, int]:
        # FIXME[refactor]: we need to restructure this class and come up with
        # a better processing scheme to subclass the Tool class ...
        # pylint: disable=arguments-differ,too-many-arguments,too-many-locals
        """Perform a targeted, adversarial attack, applying the iterative
        gradient sign method.

        Parameters
        ----------
        victim:
            The victim of the attack.
        example:
            The image data to modify
        target:
            The target class

        Results
        -------
        data:
            The modified example data.
        target:
            The target class to be predicted in case the process has
            succeeded.
        predicted:
            The actual class predicted.  This may be different to `target`
            in case the process did not succeed.
        confidence:
            The confidence value for `target` class.
        step:
            The number of iterations performed before stopping. This will
            be at most `max_steps`, but it may be less in case the
            optimization goal was reached earlier.
        """
        # Do gradient ascent (True) or descent (False)
        ascent = False

        # Create a tensor to be adapted to obtain to hold the
        # adversarial example.
        example = original.clone()
        example.requires_grad_(True)

        # obtain initial logits
        logits = victim.forward(example)
        original_logits = logits.clone()

        if target is None:
            if correct is None:
                # if no target is given, choose the class with lowest
                # confidence score (can be considered the hardest task)
                target = torch.min(original_logits, 1)[1][0]
            else:
                target = correct
                ascent = True

        # create a tensor for the target class
        # pylint: disable=not-callable
        # torch bug: https://github.com/pytorch/pytorch/issues/24807
        y_target = \
            torch.tensor([target], dtype=torch.long, device=example.device)

        # the loss function
        loss = torch.nn.CrossEntropyLoss()

        step = 0
        found = False
        LOG.info("Starting iterated gradient sign attack: "
                 "epsilon=%.2f, num_steps=%d, alpha=%.2f, "
                 "example=%s, initial=%d, correct=%s, target=%s",
                 self._epsilon, self._max_steps, self._alpha,
                 example.shape, torch.argmax(logits), correct, target)
        while (step < self._max_steps) and not found:
            self._iterate(example.detach(), logits.detach(), step, target)

            # Reset gradients
            zero_gradients(example)

            # compute the loss
            loss_cal = loss(logits, y_target)
            loss_cal.backward()

            # update the example
            with torch.no_grad():
                # obtain the sign matrix of the gradient
                update = self._alpha * torch.sign(example.grad)

                if ascent:
                    # do gradient ascent (increase the loss for the
                    # target class)
                    example += update
                else:
                    # do gradient descent (reduce the loss for the
                    # target class)
                    example -= update

                # make sure that the resulting image is not too different from
                # the original input data (image)
                if self._epsilon is not None:
                    total_diff = torch.clamp(example - original,
                                             -self._epsilon, self._epsilon)
                    example.copy_(original + total_diff)

                # Clamping the values to stay in valid range
                if vmin is not None:
                    example.copy_(torch.max(example, vmin))
                if vmax is not None:
                    example.copy_(torch.min(example, vmax))

            # obtain new logits
            logits = victim.forward(example)

            # check the stop conditions
            with torch.no_grad():
                predicted = int(torch.max(logits, 1)[1][0])
                found = (predicted == target)
                if self._min_confidence is not None:
                    # compute confidence score: apply softmax
                    confidence = F.softmax(logits, dim=1)[0, target]
                    found = found and (confidence > self._min_confidence)

            step += 1
        self._iterate(example.detach().cpu(), logits.detach().cpu(),
                      step, target)

        return (example.detach(), logits.detach(),
                original_logits.detach(), step)

    def _iterate(self, adversarial, logits, step, target) -> None:
        # pylint: disable=unused-argument
        predicted = int(torch.max(logits, 1)[1][0])
        confidence = F.softmax(logits, dim=1)[0, target]
        LOG.debug("iteration: %d/%d, "
                  "predicted: %d, target: %d (confidence=%.4f)",
                  step, self._max_steps, predicted, target, confidence)
