"""InsightFace code.


Old alignment code (MTCNN-Tensorflow)
-------------------------------------

The old alignment code be accessed in the insightface repository by typing:

```sh
git checkout 07f6547
cd src/align/
```

(to return to the master branch, just type `git checkout master`).

The actual alignment is done using MTCNN. A tensorflow implementation
is provided in the file 'src/align/detect_face.py', the pretrained
weights are stored in det1.npy, det2.npy and det3.npy.

* The [old version of the alignment code](https://github.com/deepinsight/insightface/tree/3866cd77a6896c934b51ed39e9651b791d78bb57/src/align):
  [detect_face.py](https://raw.githubusercontent.com/deepinsight/insightface/3866cd77a6896c934b51ed39e9651b791d78bb57/src/align/detect_face.py) and
  [det1.npy](https://github.com/deepinsight/insightface/raw/3866cd77a6896c934b51ed39e9651b791d78bb57/src/align/det1.npy),
  [det2.npy](https://github.com/deepinsight/insightface/raw/3866cd77a6896c934b51ed39e9651b791d78bb57/src/align/det2.npy),
  [det3.npy](https://github.com/deepinsight/insightface/raw/3866cd77a6896c934b51ed39e9651b791d78bb57/src/align/det3.npy).

```python

```

The alignment is provided by `src/common/face_preprocess.py`:

```python

```
"""

from .face_preprocess import FaceAligner
