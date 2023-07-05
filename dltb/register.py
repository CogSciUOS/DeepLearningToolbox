"""Register toolbox base implementations.  Registering the
implementations speeds up the initial import by allowing to load them
on demand.

"""

from .base.implementation import Implementable


Implementable._get_implementation_registers('dltb.tool.face.detector.Detector')
Implementable.register_subclass('dltb.tool.face.detector.Detector',
                                'dltb.tool.face.mtcnn.Detector')
