import os
import numpy as np
import cv2
from ctypes import c_int, c_ubyte, c_float,c_void_p,c_char_p,Structure,POINTER,RTLD_GLOBAL,CDLL,byref

class YOLOIMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_ubyte))]

class YOLOAFILTER(Structure):
    _fields_ = [("nfilter", c_int),
                ("filters", POINTER(POINTER(c_float)))]
    

class YOLOBOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]
    

class YOLODBOX(Structure):
    _fields_ = [("left", c_float),
                ("right", c_float),
                ("top", c_float),
                ("bottom", c_float),
                ("prob", c_float)]
  
class YOLODOUT(Structure):
    _fields_ = [("boxes", POINTER(YOLODBOX)),
                ("feats", POINTER(c_float)),
                ("feat_dim", c_int),
                ("nbox", c_int)]
        
class YoloFace(object):
    def __init__(self, root_path, lib_path=None, config_path=None, weight_path=None, afilter_path=None):       
    
        lib_path = os.path.join(root_path, 'libdarknet.so') if lib_path is None else lib_path      
        config_path = os.path.join(root_path, 'cfg', 'yolo-face.cfg') if config_path is None else config_path
        weight_path = os.path.join(root_path, 'yolo-face.weights') if weight_path is None else weight_path
        afilter_path =  os.path.join(root_path, 'anomaly_filters') if afilter_path is None else afilter_path
        
        yololib = CDLL(lib_path, RTLD_GLOBAL)
        
        self.free_anomaly_filter = yololib.free_anomaly_filter
        self.free_anomaly_filter.argtypes = [YOLOAFILTER]
        self.free_anomaly_filter.restype = None
        
        self.free_network = yololib.free_network_p
        self.free_network.argtypes = [c_void_p]
        self.free_network.restype = None
        
        self.free_detection_output = yololib.free_detection_output
        self.free_detection_output.argtypes = [YOLODOUT]
        self.free_detection_output.restype = None
        
        self.load_anomaly_filter = yololib.load_anomaly_filter
        self.load_anomaly_filter.argtypes = [c_char_p]
        self.load_anomaly_filter.restype = YOLOAFILTER
        
        self.load_network = yololib.load_YOLOv2_network
        self.load_network.argtypes = [c_char_p, c_char_p, POINTER(c_int), POINTER(c_int)]
        self.load_network.restype = c_void_p

        self.detect_face_from_image = yololib.detect_face_from_image
        self.detect_face_from_image.argtypes = [YOLOIMAGE, c_void_p, YOLOAFILTER, c_float]
        self.detect_face_from_image.restype = YOLODOUT
      
        self.extract_features_from_boxes = yololib.extract_features_from_boxes
        self.extract_features_from_boxes.argtypes = [c_int, c_int, c_void_p, c_void_p,c_void_p,c_void_p, c_void_p, c_int, c_int]
        self.extract_features_from_boxes.restype = None
        
        netw =c_int(0) 
        neth = c_int(0)
        self.net = self.load_network(config_path, weight_path, byref(netw), byref(neth))
        self.netw = netw.value
        self.neth = neth.value
        self.afilter = self.load_anomaly_filter(afilter_path)
        self.feat_dim = None

    def __del__(self):        
        self.free_anomaly_filter(self.afilter)
        self.free_network(self.net)
        
    def ndarray2YOLOIMAGE(self, raw_data):
        # default color channel order of this function is [R,G,B]
        # note that opencv is [B, G, R], so opencv needs to swap color channels
        
        result = YOLOIMAGE() 
        result.h = c_int(raw_data.shape[0])
        result.w = c_int(raw_data.shape[1])  
        result.c = c_int(raw_data.shape[2])                  
        tr_raw_data = np.transpose(raw_data, (2,0,1)) # convert (h,w,c) to (c,h,w)
        raw_flat_data = tr_raw_data.flatten()

        result.data = raw_flat_data.ctypes.data_as(POINTER(c_ubyte))
        
        # must return raw_flat_data together with result, otherwise wild pointer
        return result,raw_flat_data
      
    def box2YOLOBOX(self, raw_data):
        nYOLOBOX = YOLOBOX*len(raw_data)
        result = nYOLOBOX() 
        for ibox in range(len(raw_data)):
            result[ibox].x = (raw_data[ibox]['left'] + raw_data[ibox]['right'])/2 + self.left_offset
            result[ibox].y = (raw_data[ibox]['top'] + raw_data[ibox]['bottom'])/2 + self.top_offset
            result[ibox].w = raw_data[ibox]['right'] - raw_data[ibox]['left']
            result[ibox].h = raw_data[ibox]['bottom'] - raw_data[ibox]['top']
        
        return result

    def yolo_detect_face(self, img, thresh, use_afilter=1, gpu_id=0):    
        # input data should be unsigned int [0, 255]
        [new_h, new_w, _] = img.shape
        [net_h, net_w] = [self.neth, self.netw]
        
        self.top_offset = (net_h-new_h)/2
        self.left_offset = (net_w-new_w)/2
        embed_img= cv2.copyMakeBorder(img, self.top_offset, self.top_offset, self.left_offset, self.left_offset, 
                                cv2.BORDER_CONSTANT, value=(128,128,128))
        
        #ensure that the size of embedded image is the same as network input size
        assert embed_img.shape[0]==net_h and embed_img.shape[1]==net_w
        
        im,_ = self.ndarray2YOLOIMAGE(embed_img)
                
        cthresh = c_float(thresh)
        cuse_afilter = c_int(use_afilter)
        cgpu_id = c_int(gpu_id)
        
        detection_output =  self.detect_face_from_image(im, self.net, self.afilter, cthresh, cuse_afilter, cgpu_id)
          
        boxes = []
        for ibox in range(detection_output.nbox):
            box = detection_output.boxes[ibox]
            boxes.append({'left':box.left-self.left_offset, 'right':box.right-self.left_offset, 
                          'top':box.top-self.top_offset, 'bottom':box.bottom-self.top_offset, 'prob':box.prob})
   
        feats = np.ctypeslib.as_array(detection_output.feats, shape=(detection_output.nbox*detection_output.feat_dim,)).copy()
        feats = np.reshape(feats,(detection_output.nbox,detection_output.feat_dim))
        
        if self.feat_dim is None:
            self.feat_dim = detection_output.feat_dim
        
        # free memory
        self.free_detection_output(detection_output)
                
        return boxes,feats            
      
    def yolo_check_boxes(self,  raw_boxes, relative_flag = 0):
            
        assert(self.feat_dim is not None)
        nbox = len(raw_boxes)
        cboxes = self.box2YOLOBOX(raw_boxes)
        
        cfeats = (c_float*(nbox*self.feat_dim))()
        cconf_scores = (c_float*nbox)()
        ccorrect_coords = (c_float*(nbox*4))()
        self.extract_features_from_boxes(self.neth, self.netw, byref(cfeats),  byref(cconf_scores ), byref(ccorrect_coords), self.net, byref(cboxes), nbox, relative_flag)
        
        feats = np.ctypeslib.as_array(cfeats, shape=(nbox*self.feat_dim,)).copy()
        feats = np.reshape(feats,(nbox, self.feat_dim))        
        conf_scores = np.ctypeslib.as_array(cconf_scores, shape=(nbox,)).copy()
        correct_coords = np.ctypeslib.as_array(ccorrect_coords, shape=(nbox*4,)).copy()
        correct_coords = np.reshape(correct_coords,(nbox, 4))
        for correct_coord in correct_coords:
            correct_coord -= [self.left_offset, self.top_offset, self.left_offset, self.top_offset] 
        
        return conf_scores, correct_coords, feats
