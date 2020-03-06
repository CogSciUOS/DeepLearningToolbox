import sys
import importlib
from types import ModuleType
import inspect

# Notice that there is an issue with overwriting
# __getattr__ / __setattr__ / __getattribute__ on module level, as the
# mechanism was changed several times during the development of python:
#
# - starting from Python 3.7 it should (again) be possible to provide
#   __getattr__ / __setattr__ / __getattribute__  on module level [2].
#
# - starting from ? it is possible to add instances of arbitrary classes
#   to sys.modules during import, and the import mechanism will end
#   report that value from sys.modules.
#
#   Idea: We transform modules using lazy_import into classes using some
#   import hook. Notice however, that the post import hooks [PEP 369]
#   have been dropped in Python 3.3, as they do not work with the then
#   introduced importlib.
#
#   See [1] for details.
#
# - historic versions (?): it seems that the original behaviour of python
#   was to support __getattr__ / __setattr__ / __getattribute__ on module
#   level.
#
# [1] https://mail.python.org/pipermail/python-ideas/2012-May/014969.html
# [2] PEP 562 -- Module __getattr__ and __dir__ 
#     https://www.python.org/dev/peps/pep-0562/
def _lazy_import(lazy_imports, name):

    module_name, what = lazy_imports[name]
    print(f"_lazy_import(lazy_imports, '{name}'): importing "
          f"{(what + ' from ' + module_name) if what else module_name}"
          f" as {name})")
    module = importlib.import_module(module_name)
    # Note: if the imported module is a submodule, it may access attributes
    # of the parent module. If these are lazy attributes, then this
    # may in turn invoke _lazy_imort again: we should make sure to
    # avoid circles!

    # clean up
    del lazy_imports[name]

    return module if what is None else getattr(module, name)



#sys.modules[__name__] = Wrapper(sys.modules[__name__])

class LazyModule:
    # FIXME[todo]: could also be implemented as wrapper:
    # def __init__(self, wrapped):
    #     self.wrapped = wrapped
    # def __getattr__(self, name):
    #     Perform custom logic here
    #     try:
    #        return getattr(self.wrapped, name)
    #     except AttributeError:
    #        return 'default' # Some sensible default
    
    def __init__(self, module):
        self.module = module
        if hasattr(module, '_lazy_imports'):
            self._lazy_imports = module._lazy_imports
            del module._lazy_imports

    def __getattr__(self, name):
        try:
            attribute = getattr(self.module, name)
            print(f"LazyModule.__getattr__('{name}'): direct")
        except AttributeError:
            if name.startswith('_lazy'):
                raise AttributeError(f"No lazy attribute '{name}'")

            lazy_imports = getattr(self, '_lazy_imports', None)
            if lazy_imports is None:
                raise AttributeError(f"Trying lazy import of '{name}' "
                                     "without initialization.")

            if not name in lazy_imports:
                raise AttributeError(f"Attribute '{name}' was not registered "
                                     "for lazy import.")

            attribute = _lazy_import(lazy_imports, name)

            setattr(self.module, name, attribute)
            print(f"LazyModule.__getattr__('{name}'): lazy")

            # clean up
            if not lazy_imports:
                del self._lazy_imports
                sys.modules[self.module.__name__] = self.module

        return attribute

def _lazy_getattr(name):
    print(f"_lazy_getattr({name}): lazy importing '{name}' ...")
    frame = sys._getframe(2)
    f_globals = frame.f_globals
    if not '_lazy_imports' in f_globals:
        raise AttributeError(f"Trying lazy import of '{name}' "
                             "without initialization.")
    
    latzy_imports = f_globals['_lazy_imports']
    f_globals[name] = _lazy_import(self._lazy_imports, name)

    # clean up
    if not imports:
        del f_globals['_lazy_imports']
        lazy_getattr = pop(f_globals, None)
        if lazy_getattr is not None:
            f_globals['__getattr__'] = lazy_getattr
        else:
            del f_globals['__getattr__']
    return attribute

# FIXME[todo]: along with the module getattr you may also define a
# __dir__ function at module level to respond to dir(my_module). See
# PEP 562 for details.

def lazy_import(module, what=None, name=None):

    frame = sys._getframe(1)
    f_globals = frame.f_globals
    print(f"lazy import {module}, {what}, {name}: {frame}")

    # check if lazy import for this module has not yet been initialized
    if not '_lazy_imports' in f_globals:
        print(f"Initializing lazy import for module {frame}")
        f_globals['_lazy_imports'] = {}
        if '__getattr__' in f_globals:
            f_globals['_lazy_getattr'] = f_globals['__getattr__']
        f_globals['__getattr__'] = _lazy_getattr

    # register name(s) for lazy import
    imports = f_globals['_lazy_imports']
    if what is None:
        imports[name or module] = (module, None)
        print(f"lazy_import: {(module, None)} as {name or module}")
    elif isinstance(what, str):
        imports[name or what] = (module, what)
        print(f"lazy_import: {(module, what)} as {name or what}")
    elif isinstance(what, list):
        for _what in what.items():
            imports[_what] = (module, _what)
            print(f"lazy_import: {(module, _what)} as {_what}")
    elif isinstance(what, dict):
        if name is not None:
            raise ValueError("Both a name dictionary and a name ('{name}') "
                             "where given for or lazy import.")
        for _what, _name in what.items():
            imports[_name or _what] = (module, _what)
            print(f"lazy_import: {(module, _what)} as {_name or _what}")
    else:
        raise ValueError(f"Illegal value ({what}) for argument what.")

def lazy_register(module):
    if isinstance(module, LazyModule):
        return  # module is already registered

    if not isinstance(module, ModuleType):
        # or using inspect: inspect.ismodule(module))
        raise TypeError("Module should be a Python module, "
                        f"not {type(module)}.")

    if not hasattr(module, '__name__'):
        raise RuntimeError("Could not determine name of module.")

    name = module.__name__
    if not name in sys.modules:
        raise RuntimeError(f"Module '{name}' not registered in sys.modules.")

    module = LazyModule(sys.modules[name])
    sys.modules[name] = module
    return module

import builtins

_lazy_buildins_import = None

def _lazy_import(name, globals=None, locals=None, fromlist=(), level=0):
    #print(args, kwargs)
    print(f"_lazy_import: lazy importing '{name}': {fromlist} (level={level})...")
    result = _lazy_buildins_import(name, globals, locals, fromlist, level)
    print(f"_lazy_import: ... finished lazy importing '{name}'")
    #print(result)
    return result

def lazy_begin():
    global _lazy_buildins_import
    if _lazy_buildins_import is not None:
        raise RuntimeException("Lazy import was already started")
    _lazy_buildins_import = builtins.__import__
    builtins.__import__ = _lazy_import

def lazy_end():
    global _lazy_buildins_import
    if _lazy_buildins_import is None:
        raise RuntimeException("No lazy import that could be stopped")
    builtins.__import__ = _lazy_buildins_import
    _lazy_buildins_import = None


