"""

The module can be run as

```
python -m dltb.thirdparty.insightface --align
```


"""
from argparse import ArgumentParser
import os

# When this script is run as `python dltb/thirdparty/insightface`, the
# package name will be initialized incorrectly and relative imports
# will not work (also sys.path will contain
# dltb/thirdparty/insightface, wich may cause further problems. Hence
# we make sure, that everything is ok.

assert os.path.isabs(__file__), \
    "Run program as: python -m dltb.thirdparty.insightface"

from . import FaceAligner


def main():
    """Main program.
    """
    parser = ArgumentParser(description="InsightFace script.")
    group1 = parser.add_argument_group("Commands")
    group1.add_argument('--align', action='store_true', default=False,
                        help="align the given input file(s)")
    parser.add_argument('images', metavar='IMAGE', type=str, nargs='+',
                        help="images to process")
    args = parser.parse_args()

    if args.align:
        for image in args.images:
            name, suffix = image.rsplit('.', maxsplit=1)
            output_name = f"{name}-aligned.{suffix}"
            print(f"Aligning '{image}' to '{output_name}'.")
            print(FaceAligner)
            aligner = FaceAligner()
            img_preprocessed = aligner.preprocess(image, margin=0)
            from dltb.util.image import imshow
            imshow(img_preprocessed)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted. Good bye!")
