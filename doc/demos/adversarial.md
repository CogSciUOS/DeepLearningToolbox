# Status: transition

I am currently in the process of migrating this code into the Toolbox:
-> demos/dl-adversarial.py
-> dltb/thirdparty/torch/   (from mynet.torch)


# Adversarial examples

The demo script `demos/dl-adversarial.py` allows to generate, analyze,
and display adversarial examples.



# Code based on the tutorial: Generating Adversarial Examples using PyTorch

See: 
 * https://savan77.github.io/blog/imagenet_adv_examples.html
 * LEARNING/notebooks/adversarial/torch-adversarial-examples-imagenet.ipynb

## Requirements

### Third party dependencies

* 

# Code organization

```
├── mynet
│   ├── adversarial
│   │   ├── __pycache__
│   │   │   ├── visualization.cpython-36.pyc
│   │   │   └── visualization.cpython-37.pyc
│   │   └── visualization.py
│   ├── datasets
│   │   ├── imagenet.py
│   │   └── __pycache__
│   │       ├── imagenet.cpython-36.pyc
│   │       └── imagenet.cpython-37.pyc
│   ├── __init__.py
│   ├── model.py
│   └── torch.py
├── mynetwork.py
├── pytorch
│   ├── adex-0.jpg
│   ├── adex-100.jpg
│   ├── adex-10.jpg
│   ├── adex-1.jpg
│   ├── adex-2.jpg
│   ├── adex-3.jpg
│   ├── adex-4.jpg
│   ├── adex-5.jpg
│   ├── adex-7.jpg
│   ├── adex-8.jpg
│   ├── adex-9.jpg
│   ├── alexnet.py
│   ├── ex.jpg
│   ├── mnist.py
│   └── mytorch.py
└── README.md
```

## The file `mynetwork.py`



# Demos (to do ...)

python demos/dl-adversarial.py



# Demos (currently)

cd experiments/savan77


## 1) classify image (using default model):

```sh
python demos/dl-adversarial.py examples/cat.jpg
```

The image `examples/cat.jpg` is the file `ex4.jpg` from the
tutorial [Generating Adversarial Examples using PyTorch](https://savan77.github.io/blog/imagenet_adv_examples.html).

Using the pretrained `inception_v3` torch model, this image should be
classified as "tiger cat" (class 282), with confidence=77.3277%:
```
  1). Label: ['n02123159', 'tiger_cat'] (282), confidence=77.33%
  2). Label: ['n02124075', 'Egyptian_cat'] (285), confidence=8.41%
  3). Label: ['n02123045', 'tabby'] (281), confidence=6.62%
  4). Label: ['n02971356', 'carton'] (478), confidence=0.69%
  5). Label: ['n02127052', 'lynx'] (287), confidence=0.28%
```

## 2) perform targeted adversarial attack

```sh
python demos/dl-adversarial.py --adversarial --targets=1,5,7 examples/cat.jpg
```

## 3) perform targeted adversarial attacks on the grid (as array job):

   qsub -t 1-100:20 mynetwork.py --adversarial --targets=sge-task myimage.png


