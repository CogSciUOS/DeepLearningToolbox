from os.path import dirname, join
import sys

BASE_DIRECTORY = dirname(dirname(dirname(__file__)))
MODELS_DIRECTORY = join(BASE_DIRECTORY, 'models')

BASE_DIRECTORY in sys.path or sys.path.insert(1,BASE_DIRECTORY)
