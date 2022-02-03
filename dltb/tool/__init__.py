"""Tools for the toolbox.

"""
from .tool import Tool, Context
from .worker import Worker

Tool.register_instance('haar', 'dltb.thirdparty.opencv.face', 'DetectorHaar')
Tool.register_instance('ssd', 'dltb.thirdparty.opencv.face', 'DetectorSSD')
Tool.register_instance('hog', 'dltb.thirdparty.dlib', 'DetectorHOG')
Tool.register_instance('cnn', 'dltb.thirdparty.dlib', 'DetectorCNN')
Tool.register_instance('mtcnn', 'dltb.thirdparty.mtcnn', 'DetectorMTCNN')
