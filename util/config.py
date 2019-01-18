# https://docs.python.org/3/reference/datamodel.html#customizing-class-creation

class Config:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    @classmethod
    def _init_attributes(cls):
        pass

class MyConfig(Config):
    @classmethod
    def _init_attributes(cls):
        
