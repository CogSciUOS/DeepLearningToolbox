# Videos und Webcam

The Deep Learning ToolBox supports working with video stream.


```
    Video [Sized]
      - size
      - sequence
      - iter
      
    TimedVideo [Timed]
      - frames_per_second (fps)
      - duration (in seconds - float)
      - time_to_frame
      - frame_to_time
      - index[time index]
      
      - frame_at
      - image_at
      - frame_to_image
      
  
   
   VideoReader
       - Iterator

   Webcam (is a VideoReader)

   RewindableReader

   VideoFile (is a RewindableReader)
   
   DirectoryVideo (is a RewindableReader)
   
```

* `BufferedReader`
* `Queue`
* `AsyncReader`

## Player

A `Player` can be used to play a `Video`:
* it has properties `image` and `frame` holding the currenty
  played video frame.
* it has properties `index` and `position` holding the index/position
  of the currently played video frame.
* it provides methods `start`/`resume` and `pause`/`stop` for 
  controlling playback
* there is also a `play` method - but what is its purpose?
* it is observable, notifying interested observers whenever a new 
  frame (image) is played.
* the `Player` can be run in blocking or non-blocking mode:
  in blocking mode execution of the current thread is blocked until
  playback has finished.  In non-blocking mode, playback is done
  in some other thread and the function immediatly returns after
  initiating playback.  The caller has to observe the Player if
  interested when playback has finished.


A `TimedPlayer` has some additional means for synchronization. It
tries to ensure that playback is performed at a desired frame rate (if
possible).  If the `Video` is a `TimedVideo`, the play will aim at
achieving a suitable playback speed (which may result in skipping some
frames, or showing them more then once, if the playback rate is higher
than the source rate.  It is also possible to deviate from the
original frame rate to achieve faster or slower playback.  A
`TimedPlayer` can also be set to a specific position in time for
playback.


The `IndexTiming` class is a utility to help converting between
integer indices and different time formats.


## Reading videos


The abstract interface `dltb.base.video.Reader` defines different ways
to access a video.

```python
from dltb.base.video import Reader
from dltb.base.image import ImageDisplay

with ImageDisplay() as display:
    for frame in Reader('test.mp4'):
        display.show(frame, blocking=None)
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




## Demos


### Speed measurement



### Watching imagesources

```python
from dltb.util.video import show
from dltb.datasource import Thumbcinema
import datasource.predefined

show(Thumbcinema('imagenet-val'))
```
