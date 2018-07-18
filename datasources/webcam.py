from datasources import DataSource, InputData
import cv2

class DataWebcam(DataSource):
    '''A data source fetching images from the webcam.

    Attributes
    ----------
    _capture    :   
                    A capture object
    '''
    _capture  = None

    def __init__(self):
        '''Create a new DataWebcam
        '''
        super().__init__()
        print(cv2.__file__)
        self._capture = cv2.VideoCapture(0)

    def __getitem__(self, index):
        ret, frame = self._capture.read()
        return InputData(frame, "Webcam")
    
    def __len__(self):
        # FIXME[hack]
        return 100
