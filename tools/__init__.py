from .tool import Tool, Processor

Tool.register_key('haar', 'tools.face.opencv', 'DetectorHaar')
Tool.register_key('ssd', 'tools.face.opencv', 'DetectorSSD')
Tool.register_key('hog', 'tools.face.dlib', 'DetectorHOG')
Tool.register_key('cnn', 'tools.face.dlib', 'DetectorCNN')
Tool.register_key('mtcnn', 'tools.face.mtcnn', 'DetectorMTCNN')

Tool.add_module_requirement('tools.face.mtcnn', 'mtcnn')
Tool.add_module_requirement('tools.face.opencv', 'cv2')
Tool.add_module_requirement('tools.face.dlib', 'dlib')
