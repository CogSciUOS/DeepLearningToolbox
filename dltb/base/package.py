"""General code for managing packages.
"""

# standard imports
from typing import Union, Optional
from types import ModuleType
from pathlib import Path
import os
import sys
import logging
import subprocess

# toolbox imports
from ..util.importer import importable, import_module
from .register import RegisterClass
from .prepare import Preparable
from .info import InfoSource
from .resource import Installable

# logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)



class Package(InfoSource, Preparable, Installable, register=True,
              metaclass=RegisterClass):
    """A `Package` represents a Python package, that is a collection of
    Python modules distributed in a bundle.

    Installation
    ------------

    A Python package can be installed using a Python package manager
    like pip or conda.

    Module
    ------

    A package usually has main module.  The name of that module is
    often the same as the package name but may also deviate.  When
    using a package, one has to import the module name (not the
    package name).

    For some packages there are common aliases used when importing the
    package, like `np` for `numpy` or `tf` for `tensorflow`.  It is
    not mandatory to use these aliases, and the module can also be
    imported by its standard name or any other name.  As these aliases
    are so common, the `Package` will provide this information,
    without making explicit use of it.

    Prepration (import)
    -------------------
    Before a package can be used, it has to be imported.  Depending
    on the size of the package (number of modules included), importing
    the package may take some time.

    Importing the main module of a package will usually not import its
    submodules.  However, there are some well-known exceptions to
    rule, like for example `tensorflow` and `torch` which usually
    import all of their submodules.  Also some modules of the standard
    Python library changed their behaviour over time, which results in
    notorious `AttributeErors` when trying to access submodules, that
    were automatically imported in previous version.  As a general
    rule of thumb, one should not rely on submodules to be imported
    automatically, but rather import all required submodules
    explicitly.

    Arguments
    ---------
    module: str
        The fully qualified module name that can be used to check if
        the ``Package`` is installed. If ``None``, the ``key`` of
        this ``Resource`` will be used.
    alias: str
        An optional alias commonly used for this package, like
        `'np'` for `numpy`.
    pypi: str
        The pypi name of this package.
    conda: str
        The conda name of this package.
    conda_channel: str
        The conda channel for installing this package.

    """
    _module_name: str = None
    _alias: Optional[str] = None
    _pypi: str = None
    _conda: dict = None

    _prepare_on_init: bool = False
    
    def __init__(self, key: Optional[str] = None,
                 module: Optional[str] = None,
                 label: Optional[str] = None,
                 alias: Optional[str] = None,
                 pypi: Optional[str] = None,
                 conda: Optional[str] = None,
                 conda_channel: str = '',
                 **kwargs) -> None:
        # pylint: disable=too-many-arguments
        if key is None:
            key = module
        old_package = Package[key] if key in Package else None

        super().__init__(key=key, **kwargs)

        module = module or (old_package and old_package.module_name)
        if module is not None:
            self._module_name = module

        label = label or (old_package and old_package.label)
        if label is not None:
            self._label = label

        alias = alias or (old_package and old_package.alias)
        if alias is not None:
            self._alias = alias

        if pypi:
            self.set_pypi_source(pypi)
        if conda is not None:
            self.set_conda_source(conda, conda_channel)

    @property
    def module_name(self) -> str:
        """The name of the main module of this `Package`.
        """
        return self._module_name or self._key

    @property
    def module(self) -> Optional[ModuleType]:
        """The main module of this `Package` (if already loaded) or `None`
        (if the `Package` has not been loaded yet).
        """
        return sys.modules.get(self.module_name, None)

    @property
    def label(self) -> str:
        """A label to be used for this `Package` when displaying in a user
        interface.
        """
        return self._label or self.module_name

    @property
    def alias(self) -> str:
        """The name (alias) commonly used when importing (the main module of)
        this `Package`, e.g. `'np'` for `numpy`.
        """
        return self._alias or self.module_name

    def set_conda_source(self, name: str, channel: str = '') -> None:
        """Set the conda source allowing to install the module.

        Attributes
        ----------
        name: str
            The name of the conda package.
        channel: str
            The conda channel from which the module should
            be installed. If none is provided, the default
            channel will be used.
        """
        if self._conda is None:
            self._conda = {}
        self._conda[channel] = name

    def set_pypi_source(self, name: str) -> None:
        """Set the pypi information required for installing this package
        with pip.

        Attributes
        ----------
        name: str
            The name of the pip package.
        """
        self._pypi = name

    def _installed(self) -> bool:
        """Check if this :py:class:`ModuleResource` is installed.
        """
        return importable(self.module_name)

    def _install(self, method: str = 'auto', **kwargs):
        # pylint: disable=arguments-differ
        """Install this :py:class:`ModuleResource`

        Arguments
        ---------
        method: str
            The installation method to use. Supported are
            'pip' for installation with pip and 'conda' for
            conda installation. 'auto' tries to autoamtically
            determine the best installation method.
        """
        # Are we using conda?
        # FIXME[todo]: provide a general function for this check
        # FIXME[hack]: this does only work in conda environments,
        # but not if the conda base environment is used
        if method == 'conda' or ('CONDA_PREFIX' in os.environ and self._conda):
            self._conda_install()
        else:
            self._pip_install()

    def _pip_install(self) -> None:
        """Install package via pip.
        """
        # use pip
        #

    def _conda_install(self) -> None:
        """Install package with conda.
        """
        # FIXME[todo]: there is a python 'conda' module
        # which may be used ...

        command = ['conda', 'install']

        channel = next(self._conda)
        if channel:
            command += '-c', channel
        command.append(self._conda[channel])
        with subprocess.Popen(command,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as process:
            _stdout, _stderr = process.communicate()
        #stdout, stderr

    def _preparable(self) -> bool:
        """Check if the :py:class:`Package` is preparable, i.e., if the
        package is installed and can be imported.

        """
        return self.installed

    def _prepared(self) -> bool:
        """Check, if this :py:class:`ModuleResource` was prepared,
        that is, if the module was imported.

        Returns
        ------
        True if the module was already imported, False otherwise.
        """
        return self._module_name in sys.modules

    def _prepare(self):
        """Prepare this :py:class:`Package`, that is, import the central
        module of the package.

        Importing a module may take some time, especially for modules
        from larger packages.  It can also fail, for example if the
        (package containing the) module is not installed, or in case
        of some misconfiguration or error in that module.
        """
        import_module(self.module_name)

    def unprepare(self) -> None:
        """Unpreparin a package means to unload the corresponding module(s)).
        This is currently not implemented.
        """
        raise NotImplementedError("Unloading modules not implemented yet.")

    @property
    def version(self) -> str:
        """A generic method to determine the package version.  Notice that
        this method may fail, for example if a package was not
        imported yet, or it does not provide a standard way to obtain
        version information. In that case, the version will be a
        message indicating the problem.
        """
        module = self.module
        if module is None:
            return "not loaded"
        if hasattr(module, '__version__'):
            return module.__version__
        return "loaded, no version"

    @property
    def directory(self) -> Optional[Path]:
        """The directory into which the package is installed.  May be `None`
        if the directory can not be determined or the package is not
        installed yet.
        """
        module = self.module
        if module is None:
            return None
        if hasattr(module, '__file__'):
            return Path(module.__file__).parents[0]
        return None

    def initialize_info(self) -> None:
        # we add this dynamical information here - to reflect a change
        # of state.  Probably it would be more efficient to register
        # hooks that updates the information on install/import ...
        self.add_info('module', self.module_name,
                      title='Module')
        self.add_info('package_installed', self._installed,
                      title='Package installed')
        self.add_info('package_loaded', lambda: self.prepared,
                      title='Package loaded')
        self.add_info('version', lambda: self.version,
                      title='Version')


Packagelike = Union[Package, str]
