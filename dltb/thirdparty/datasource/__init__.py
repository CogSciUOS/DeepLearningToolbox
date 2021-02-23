"""Predefined Datasources.
"""

# toolbox imports
from ...datasource import Datasource

Datasource.register_instance('imagenet-val', __name__ + '.imagenet',
                             'ImageNet', section='val')  # section='train'
Datasource.register_instance('dogsandcats', __name__ + '.dogsandcats',
                             'DogsAndCats')
Datasource.register_instance('widerface', __name__ + '.widerface', 'WiderFace')
Datasource.register_instance('fgnet', __name__ + '.fgnet', 'FGNet')
Datasource.register_instance('Helen', __name__ + '.helen', 'Helen')
Datasource.register_instance('lfw', __name__ + '.lfw', 'LabeledFacesInTheWild')
Datasource.register_instance('ms-celeb-1m', __name__ + '.face', 'MSCeleb1M')
Datasource.register_instance('5celeb', __name__ + '.fivecelebface',
                             'FiveCelebFace')

Datasource.register_class('WiderFace', __name__ + '.widerface')
