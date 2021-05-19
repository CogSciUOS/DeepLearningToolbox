"""A module providing basic highscore functionality.
"""

# standard imports
from typing import Union, Tuple, Sequence, Iterator, BinaryIO
from abc import ABC, abstractmethod

# third party imports
import numpy as np

# toolbox imports
from ..util import nphelper

Discipline = Tuple
Owner = Tuple
Score = float


class HighscoreBase:
    """A :py:class:`Highscore` is a ranked list of best values in some
    category (the top n). This list entry contains the actual score
    and the owner of that score.

    Example: an activation highscore can record the top activation
    vaues for individual features in a network. The different
    features, identified by network, layer, and channel, constitute
    the disciplines, while identifiers should allow to input stimulus
    (e.g. a filename or an index in a :py:class:`Datasource`), and
    potentially a location in a feature map.

    """

    def __init__(self, top: int = 10,
                 owner_dimensions: Union[int, None] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._top = top
        self._owner_dimensions = owner_dimensions

    def __len__(self) -> int:
        return self._top

    @property
    def owner_dimensions(self) -> Union[int, None]:
        """The number of indices required to identify an owner. If `None`,
        a single value is sufficient and will be used. If an integer
        (including `1`), owners will be represented as tuples of that
        length.
        """
        return self._owner_dimensions


class Highscore(HighscoreBase, ABC):
    """Abstract base class for a highscore.  A highscore provides a
    sequence of scores and a corresponding sequence of owners of these
    scores.
    """

    @property
    @abstractmethod
    def scores(self) -> Sequence[Score]:
        """The sequence of score values of this highscore. Corresponding
        owners can be obtained from :py:prop:`owners`.
        """

    @property
    @abstractmethod
    def owners(self) -> Sequence[Owner]:
        """The sequence of owners of the score values provided by
        :py:prop:`scores`.
        """


class HighscoreGroup(HighscoreBase, ABC):
    """A :py:class:`HighscoreGroup` is a group of related
    :py:class:`Highscore`s. These can be stored and updated
    simultanously, meaning that they use the same scheme to identify
    owners.

    """
    class View(Highscore):
        """A view on a single :py:class:`Highscore` of a
        :py:class:`HighscoreGroup`.
        """

        def __init__(self, group: 'HighscoreGroup', index: int) -> None:
            super().__init__(top=len(group))
            self._scores = group.scores(index)
            self._owners = group.owners(index)

        @property
        def scores(self) -> Sequence[Score]:
            """The sequence of score values of this highscore. Corresponding
            owners can be obtained from :py:prop:`owners`.
            """
            return self._scores

        @property
        def owners(self) -> Sequence[Owner]:
            """The sequence of owners of the score values provided by
            :py:prop:`scores`.
            """
            return self._owners

    def __init__(self, size: int = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if size is None:
            raise ValueError("A positive group size is expected.")
        self._size = size

    @property
    def size(self) -> int:
        """The group size, that is the number of highscores grouped in
        this :py:class:`HighscoreGroup`
        """
        return self._size

    def __getitem__(self, index: int) -> Highscore:
        return HighscoreGroup.View(self, index)

    def scores(self, index: int) -> Sequence[Score]:
        """The scores for the group member `index`.
        """
        return self[index].scores

    def owners(self, index: int) -> Sequence[Owner]:
        """The score ownsers for the group member `index`.
        """
        return self[index].owners


class HighscoreCollection(HighscoreBase, ABC):
    """A :py:class:`HighscoreCollection` can hold Highscores (and
    HighscoreGroups) for multiple disciplines, allowing to record
    different scores in each discipline.

    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._highscores = dict()

    @abstractmethod
    def disciplines(self) -> Iterator[Discipline]:
        """The disciplines
        """

    @abstractmethod
    def highscore(self, discipline: Discipline) -> Highscore:
        """Get the highscore, that is the list of top scores and their
        owners, for the given discipline.
        """

    def scores(self, discipline: Discipline) -> Sequence[Score]:
        """The score values for the given discipline.
        """
        return self.highscore(discipline).scores

    def owners(self, discipline: Discipline) -> Sequence[Owner]:
        """The high score owners for the given discipline.
        """
        return self.highscore(discipline).owners


class HighscoreGroupNumpy(HighscoreGroup):
    """A :py:class:`HighscoreGroup` realized using Numpy. A group of
    highscores is implemented by a pair of `np.ndarray`, one holding
    the scores and one the associated owners.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # owners: (size, top) or (size, top, owner_dimensions)
        owners_shape = (self.size, len(self))
        if self.owner_dimensions is not None:
            owners_shape += (self.owner_dimensions, )
        self._owners = np.full(owners_shape, -1, np.int)
        # scores: (size, top)
        self._scores = np.full((self.size, len(self)), np.NINF, np.float32)

    def scores(self, index: int) -> Sequence[Score]:
        """The score values of the group member with the given `index`.
        """
        return self._scores[index]

    def owners(self, index: int) -> Sequence[Owner]:
        """The score owners for the group member with the given `index`.
        """
        return self._owners[index]

    def update(self, owners: np.ndarray, scores: np.ndarray) -> None:
        """Update this :py:class:`HighscoreGroupNumpy` with new
        values.

        Arguments
        ---------
        owners:
            The owners of the new scores. This is an array of dtype int and
            shape (batch_size, owner_dimensions)
        scores:
            The new scores. This is an array of dtype float and
            shape (batch_size, group_size).
            It is also possible to provide score values that do not
            only have batch dimension, but also some addition index
            axis, that is, they have the shape
            (batch_size, indices..., group_size).
            The indices will be used as part of the owner specification.
            That is score[batch, pos_x, pos_y] will be indexed as
            (owners[batch], pos_x, pos_y).
        """
        #
        # step 1: get the top scores and corresponding owners
        #
        top = min(len(self), len(scores))

        # Compute top_scores of shape (group_size, top)
        # and top_owners of shape (group_size, top, owner_dimensions)
        if scores.ndim > 2:

            # slim_scores have shape (batch_size*indices..., group_size)
            slim_scores = scores.reshape((-1, scores.shape[-1]))

            # top_slim: array of shape (top, group_size),
            #   containing the indices in the slim_scores array for the top
            #   elements for each highscore (i.e. values from 0 to size)
            top = min(self._top, len(slim_scores))
            top_slim = nphelper.argmultimax(slim_scores, num=top, axis=0)

            # top_scores: (group_size, top)
            top_scores = np.take_along_axis(slim_scores, top_slim, axis=0).T

            #
            # obtain the top owners.
            #

            # (a) the owner_shape (batch_size, indices...) describes the
            #     coordinates required to identify an owner. This is a global
            #     index and some optional local indices. This is essentially
            #     score.shape without the group_size axis.
            owner_shape = scores.shape[:-1]

            # (b) compute first version of the top_owners. These will have the
            #     shape (group_size, top, len(owner_shape)), because:
            #     - np.unravel_index produces len(aux_shape) index arrays,
            #       each with top_slim.shape, that is (top, group_size).
            #     - stack combines these to an array of shape
            #       (len(aux_shape), top, group_size)
            #     - transposition yields the final array.
            #     top_owners[member, n, 0] contains the batch index
            #     top_owners[member, n, 1] contains the first local index
            #     top_owners[member, n, 2] contains the second local index
            #     ...
            top_owners = np.stack(np.unravel_index(top_slim, owner_shape)).T

            # (c) the first owner index still contains the batch index,
            #     not the globl owner coordinate. If an owner array was
            #     provided (and is one-dimensional), we can use it to map
            #     the batch index to the owner coordinate.
            if owners is not None and owners.shape[1] == 1:
                top_owners[:, :, 0] = owners[top_owners[:, :, 0]]

        elif top > len(self):  # consider only new top scores
            # top_indices: (group_size, top)
            top_indices = nphelper.argmultimax(scores, num=top, axis=0).T

            # top_scores: (group_size, top)
            top_scores = np.take_along_axis(scores.T, top_indices, axis=1)

            # top_owners: (group_size, top, owner_dimensions)
            top_owners = owners[top_indices]

        else:  # top <= len(self): consider all new scores
            # top_scores: (top, group_size) -> (group_size, top)
            top_scores = scores.T
            # top_owners:
            #   (top, owner_dimensions) -> (size, top, owner_dimensions)
            top_owners = owners[np.newaxis].repeat(self.size, axis=0)

        #
        # 2. join current and new scores
        #
        joint_owners = np.append(self._owners, top_owners, axis=1)
        joint_scores = np.append(self._scores, top_scores, axis=1)

        #
        # 3. get new top elements from joint scores
        #
        indices = \
            nphelper.argmultimax(joint_scores, len(self), axis=1, sort=True)
        self._scores[:] = np.take_along_axis(joint_scores, indices, axis=1)
        for member in range(self.size):
            self._owners[member] = joint_owners[member, indices[member]]

    def store(self, outfile: BinaryIO) -> None:
        """Store the current state of this :py:class:`HighscoreGroup`.
        """
        np.save(outfile, self._owners)
        np.save(outfile, self._scores)

    def restore(self, infile: BinaryIO) -> None:
        """Restore the state of this :py:class:`HighscoreGroup` from
        a file like object.
        """
        self._owners = np.load(infile)
        self._scores = np.load(infile)
