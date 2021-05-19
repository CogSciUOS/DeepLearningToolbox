""":py:class:`Storable` objects allow to store their state to restore
it later.

"""

# standard imports
from typing import BinaryIO, Union
from pathlib import Path
import os
import json
import logging

# toolbox imports
from .prepare import Preparable

# logging
LOG = logging.getLogger(__name__)


class Storable(Preparable):
    """A :py:class:`Storable` object provides means to store its state
    persistently and restore it later.

    A typical application is to allow to abort and resume a work
    process. For this to work, the current state of the process has to
    be stored.

    Another application is to store results of complex computations
    for later inspection by another tool.  This can be a faster
    alternative to directly executing the computations when values are
    needed.

    Arguments
    ---------
    store: bool
        A flag indicating that the persistent work mode should be applied,
        that is that the state of the object should automatically be stored
        after usage. Storage is performed upon unpreparing the object, which
        can be initiated explicitly by calling :py:meth:`unprepare` or
        implicitly, for example by deleting the object.
    restore: bool
        A flag indicating that the state of the object should be restored.
        The default behaviour (indicated by `None`) is to try restoration
        if the `store` flag is `True` but to not complain (raise exception)
        if restoration fails.  If explicitly set to `False`, no restoration
        is tried, even if `store` is `True`. This allows to overwrite
        an existing stored state. If explicitly set to `True`, preparation
        of the object fails if restoration raises an exception.

    These two flags allow to realize different scenarios:

    Scenario 0 - no store (store=False, restore=False/None, default):
        the object neither initializes from a stored
        state, nor does it store its state upon deletion.

    Scenario 1 - persistent (store=True, restore=None):
        the object state is restored upon initialization (that is
        during preparation) if a stored state exist, and stores its
        state upon deletion (that is during unprepare)

    Scenario 2 - continue (store=True, restore=True)
        like scenario 1 - but initialization (prepare) fails if no
        stored state exists

    Scenario 4 - overwrite (store=True, restore=False)
        like scenario 1 - but alwas start with a fresh state, even
        if a stored state exists (such a state will be overwritten
        upon deletion of the object).

    Scenario 5 - read only (store=False, restore=True)
        initialize object from a stored state but do not store changes
        upon deletion. If no stored state exists, an exception is thrown


    Subclassing
    -----------

    There are two ways to make classes :py:class:`Storable`: (1) by
    providing a collection of storable attributes and (2) by providing
    store and restore methods. The methods can be used individually or
    in combination.

    (1) Subclasses of the :py:class:`Storable` can introduce an
    optional `storables` argument to the class definition.  This should
    provide a list of names for attributes to be stored.

    (2) Subclasses can implement the methods :py:meth:`_store` and
    :py:meth:`_restore`. Within this method, they can perform custom
    steps for storing and restoring (=initializing object from storage).

    The general goal is to be agnostic concerning the storage
    mechanism (file, database, ...). However, currently the emphasis
    is on file storage (using a :py:class:`FileStorage` object
    providing some configuration and auxiliary functions).  There is
    one central file into which the storable attributes are stored.
    Additional data can be stored in this file as well. This can
    be achieved by implementing the methods :py:meth:`_store_to_file`
    and :py:meth:`_restore_from_file`, which will get a file handle
    as argument.

    """
    _storables = set()

    def __init_subclass__(cls: type, storables: set = None, **kwargs):
        # pylint: disable=arguments-differ
        """Initialization of subclasses of :py:class:`Storable`.
        Each of these classes will provide an extra class property
        `storables`, listing the attributes that are to be stored.

        Arguments
        ---------
        storables: Sequence[str]
            A list of additional properties that should be stored for
            the new subclass. These will be added to the storables
            of superclass to obtain the actual list of storables for
            the new subclass.
        """
        super().__init_subclass__(**kwargs)
        LOG.debug("Initializing new Storable class: %s, new storables=%s",
                  cls, storables)

        new_storables = set() if storables is None else set(storables)
        for base in cls.__bases__:
            if issubclass(base, Storable):
                # pylint: disable=protected-access
                new_storables |= base._storables
        if new_storables != cls._storables:
            cls._storables = new_storables

    def __init__(self, store: bool = None, restore: bool = None,
                 directory: Union[Path, str] = None,
                 filename: Union[Path, str] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._store_flag = store
        self._restore_flag = restore
        if directory is not None or filename is not None:
            self._storage = FileStorage(filename=filename, directory=directory)
        else:
            self._storage = None

    def _prepare(self) -> None:
        super()._prepare()
        if self._restore_flag is True:
            self.restore()
        elif self._restore_flag is None and self._store_flag:
            try:
                self.restore()
            except Exception:  # pylint: disable=broad-except
                self._fresh()
        else:
            self._fresh()

    def _pre_unprepare(self) -> None:
        """Storing the state of a :py:class:`Storable` object should
        be done before resources of the object are released.
        """
        print(f"store_flag={self._store_flag}")
        if self._store_flag:
            self.store()
        super()._pre_unprepare()

    def store(self) -> None:
        """Store the current state of this :py:class:`Storable` object.
        """
        self._store()

    def _store(self) -> bool:
        """This method should be implemented by subclasses.
        """
        self._storage.store(self)

    def restore(self) -> None:
        """Restore the current state of this :py:class:`Storable` object from
        the persistent storage. This initializes (prepares) the object
        and hence will be called on an unprepared (or only partly
        prepared) object.

        """
        if not self.restorable:
            raise RuntimeError("Object is not restorable.")
        self._restore()

    def _restore(self) -> bool:
        """This method should be implemented by subclasses. It is supposed
        to prepare the object by reading in stored values.  In other
        words, this is an alternative to a fresh preparation, which
        should be done in :py:meth:`_fresh`. Code that should be
        executed in both cases (restore and fresh preparation) should
        go into the :py:meth:`_prepare` method.
        """
        self._storage.restore(self)

    def _fresh(self) -> bool:
        """This method should be implemented by subclasses. This is the
        place to perform initialization of storable properties. This
        method is only called if no restoration takes place. It will
        be called by :py:class:`Storable._prepare`.
        """

    @property
    def restorable(self) -> bool:
        """Check if this :py:class:`Storable` object can be restored.
        This is the case if a persistent storage (e.g., a file or a
        database) is available.
        """
        return self._restorable()

    def _restorable(self) -> bool:
        """This method should be implemented by subclasses.
        """
        return self._storage.exists()

    #
    # File specific storage
    #

    def store_to_file(self, outfile: BinaryIO) -> None:
        """Store this :py:class:`Storable` into a file.

        Subclasses may extend this mechanism by overriding this
        function. If doing so, the first command should be
        `super().store_to_file(outfile)`.

        Arguments
        ---------
        outfile:
            A writable filelike object.
        """
        values = {}
        for attribute in self._storables:
            values[attribute] = getattr(self, attribute)
        json.dump(values, outfile)

    def restore_from_file(self, infile: BinaryIO) -> None:
        """Restore this :py:class:`Storable` from a file.  This initializes
        (prepares) the object with data from that file, hence it will
        be called on an unprepared (or only partly prepared) object.

        Subclasses may extend this mechanism by overriding this
        function. If doing so, the first command should be
        `super().restore_from_file(outfile)`.

        Arguments
        ---------
        infile:
            A readable filelike object.
        """
        values = json.load(infile)
        for attribute in self._storables:
            try:
                setattr(self, attribute, values.pop(attribute))
            except KeyError:
                LOG.warning("Storable: attribute '%s' missing in file %s.",
                            attribute, infile)
        if values:
            LOG.warning("Storable: unkown attributes in file %s: %s",
                        infile, list(values.keys()))


class Storage:
    """An abstract class representing a storage mechanism.
    """

    def store(self, storable: Storable) -> None:
        """Store a :py:class:`Storable` in this :py:class:`Storage`.
        """

    def restore(self, storable: Storable) -> None:
        """Restore a :py:class:`Storable` in this :py:class:`Storage`.
        """


class FileStorage(Storage):
    """A storage realized by one or multiple files.
    """

    def __init__(self, directory: Union[Path, str] = None,
                 filename: Union[Path, str] = 'meta.json',
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._directory = Path(directory)
        self._filename = Path(filename)

    def __str__(self) -> str:
        return (f"FileStorage(directory={self._directory}, "
                f"filename={self._filename})")

    @property
    def directory(self) -> Path:
        """The name of the directory into which data for this
        :py:class:`FileStorage` is stored on disk.
        """
        return self._directory

    def filename(self, name: str = None) -> Path:
        """The absolute filename holding the data.
        """
        if name is None:
            if self._filename is None:
                raise ValueError("No name provided for filename")
            name = self._filename
        return self.directory / name

    def exists(self, name: str = None) -> bool:
        """Check if the fiven file exists.
        """
        return self.filename(name).exists()

    def store(self, storable: Storable) -> None:
        """Store a :py:class:`Storable` in this :py:class:`Storage`.
        """
        os.makedirs(self.directory, exist_ok=True)
        with open(self.filename(), 'w') as outfile:
            storable.store_to_file(outfile)

    def restore(self, storable: Storable) -> None:
        """Restore a :py:class:`Storable` in this :py:class:`Storage`.
        """
        with open(self.filename(), 'r') as infile:
            storable.restore_from_file(infile)
