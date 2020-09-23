"""Tools for the toolbox.

"""
from .tool import Tool

Tool.register_key('haar', 'dltb.thirdparty.opencv.face', 'DetectorHaar')
Tool.register_key('ssd', 'dltb.thirdparty.opencv.face', 'DetectorSSD')
Tool.register_key('hog', 'dltb.thirdparty.dlib', 'DetectorHOG')
Tool.register_key('cnn', 'dltb.thirdparty.dlib', 'DetectorCNN')
Tool.register_key('mtcnn', 'dltb.thirdparty.mtcnn', 'DetectorMTCNN')

Tool.add_module_requirement('dltb.thirdparty.mtcnn', 'mtcnn')
Tool.add_module_requirement('dltb.thirdparty.opencv.face', 'cv2')
Tool.add_module_requirement('dltb.thirdparty.dlib.face', 'dlib')
