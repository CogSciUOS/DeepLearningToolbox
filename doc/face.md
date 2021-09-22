# Face processing


# Face detection

The central class for face detection is
`dltb.tool.face.detector.Detector`. That class defines 
the following functions:

* `detector.detect(image: Imagelike) -> Metadata`:
  synchronous API.  The function returns a `Metadata` object,
  containing a `Region` attribute holding a list of detections.
  
* `detector.process(image: Image) -> None`: asynchronous API.
  Apply the detector to the given `Image` object and store the
  results in a detetor-specific property of that object.


* `detector.detect_boxes(image: Imagelike) -> Iterable[BoundingBox]`:
  specific interface for detectors that can report bounding boxes.
  The detections can than be accessed as `detector.detections(image)`.


```
Metadata
+--
```

## Utility functions

### Marking detections in an image

### Extracting detections from an image



## Creation

A `dltb.tool.face.detector.Detector` can be created in the following
ways:
* by expliticly instantiating one of the implementations:
```python
from dltb.thirdparty.opencv.face import DetectorHaar

detector = DetectorHaar()
```

# Landmark detection

## MTCNN

The MTCNN bounding box and landmark detector has become a very popular
tool due to its high reliability and ease of use.  There are several
implementations and pretrained models of MTCNN:

* Kaipeng Zhang (kpzhang93) from 2016:
  [MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment).
  An old Caffe implementation, also employing Matlab code.

* The `mtcnn` python modul based on a Keras implementation

* Another Tensorflow-implementation from (on old version of) the InsightFace
  project (see [thirdparty/insightface.md]).

* A torch implementation from the face.evoLVe project
  (see the `faceevolve/` subdirectory)


# Face alignment



# Face recognition



## Face verification



## Face identification



# Face generation
