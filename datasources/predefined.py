from .source import Datasource

import importlib

class Metaclass(type):

    def __getitem__(cls, id):
        if id not in Predefined.datasources:
            raise KeyError(f"Unknown datasource name {id}")
        
        datasource = Predefined.datasources[id]
        if isinstance(datasource, Datasource):
            return instance

        module_name, class_name, args, kwargs = datasource
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)
        
    def __contains__(cls, id):
        return id in Predefined.datasources

class Predefined(Datasource, metaclass=Metaclass):
    """An abstract base class for predefined data sources.
    """
    #
    # Static data and methods
    #

    datasources = {}

    @staticmethod
    def get_data_source_ids():
        return list(Predefined.datasources.keys())

    @staticmethod
    def get_data_source(id):
        return Predefined.datasources[id]

    @staticmethod
    def register_id(id, module_name, class_name, *args, **kwargs):
        if id in Predefined.datasources:
            raise ValueError("Duplcate registration of Datasource "
                             f"with ID '{id}'.")
        Predefined.datasources[id] = (module_name, class_name, args, kwargs)

    _id: str = None

    def __init__(self, id: str=None, **kwargs):
        if id is None:
            raise ValueError("You have to provde an id for "
                             "a Predefined datasoure")
        super().__init__(**kwargs)
        self._id = id
        Predefined.datasources[id] = self

    @property
    def id(self):
        """Get the "public" ID that is used to identify this datasource.  Only
        predefined Datasource should have such an ID, other
        datasources should provide None.
        """
        return self._id

    def get_public_id(self):
        """Get the "public" ID that is used to identify this datasource.  Only
        predefined Datasource should have such an ID, other
        datasources should provide None.
        """
        return self._id

    def check_availability(self):
        """Check if this Datasource is available.

        Returns
        -------
        True if the Datasource can be instantiated, False otherwise.
        """
        return False

    def download(self):
        raise NotImplementedError("Downloading this datasource is "
                                  "not implemented yet.")
