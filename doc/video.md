# Videos und Webcam

The Deep Learning ToolBox supports working with video stream.


## Reading videos


The abstract interface `dltb.base.video.Reader` defines different ways
to access a video.

```python
from dltb.base.video import Reader
from dltb.base.image import ImageDisplay

with ImageDisplay() as display:
    for frame in Reader('test.mp4'):
        display.show(frame)
```
The `frame` is 


### Accessing frames

```python
from dltb.base.video import Reader
from dltb.base.image import ImageDisplay

with ImageDisplay() as display, Reader('test.mp4') as reader:
    frame = reader[24]
    display.show(frame, timeout=3)
    
    frame = reader[2.1]
    display.show(frame, timeout=3)

    frame = reader['00:00:02.1']
    display.show(frame, timeout=3)
```


## Processing Videos


## Video Datasource


* Q: Why is `dltb.base.video.Reader` not a `Datasource`?

A: mainly to reduce dependencies. The idea is that `dltb.base.video`
(with the relevant implementations from `dltb.thirdparty`) should
be useable independent of the other parts of the Deep Learning Toolbox.

