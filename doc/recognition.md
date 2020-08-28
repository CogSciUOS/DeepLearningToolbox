# Recogntion

In image processing refers to the task of recognizing objects in an
image, that is saying what is where in the picture. The task is
usually broken down into multiple steps:
* detection: the detector locates objects of interest in the image, that
  is, it estimates position and size of the objects. Typically, a detector
  return a collection of bounding boxes.
* normalization: the depiction of objects in images can show large variation,
  e.g., due to lighting conditions, orition, viewing angle and distance,
  deformations or occlusions. The goal of normalization is to reduce
  the variance in order to obtain a standardized depiction of the object.
* classification: finally the (normalized) images of the individual objects
  are assigned to object classes. Often classification can be separated
  into a feature extraction (or embedding) part, and the actual
  classification.


## Detection

Detection is realized by the Detector class. The central function is
* detect(data) -> locations



There are different implementation of detectors, for different types
of object to detect.


### Extraction

A detector usually outputs a collection of bounding boxes. For further
processing one will usually extract the detected regions from the
original input image. There are some points to consider here:

* How to deal with points outside the image? Some detectors may return
  bounding boxes that lay partly outside the image if objects are located
  near the image boundary. Possible approaches are (1) to just use the valid
  part of the bounding box, (2) to fill the invalid part of the bounding box
  with some fixed value, or (3) to completely ignore such invalid bounding
  boxes (basically dropping the detection).

* Resizing: Often subseeding steps expect input images of fixed size.
  
* Aspect ratio: if resizing requires a change of aspect ratio, there
  are essentially two options: cropping the image to keep the original
  aspect ratio or resizing and thereby loosing the original aspect ratio.


The extractor can be provide
* extract(data, locations) -> Data[Batch]


## Alignment

Alignment refers to the process of transforming the image of an object
to show the object in some standard position. For example, in face
alignment one usually aims at getting a frontal view to an upright
face.

## Landmarking

Landmarking is the task of locating a predefined set of landmarks in a
given image. For example, in facial landmarking on might be interested
in the positions of eyes, mouth and nose. Landmarking may be used as a
preparatory step to alignment.
