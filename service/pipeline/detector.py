"""Expanded COCO observations using the existing verified Apache2.0 model."""
import threading
import cv2,numpy as np
from src import video_objects as base
from src.video_analysis import _check_deadline

CLASSES={0:'person',24:'backpack',26:'handbag',28:'suitcase',39:'bottle',41:'cup',56:'chair',60:'dining_table',62:'display',63:'laptop',67:'cell_phone',72:'refrigerator',73:'book'}
_VERSION='yolox-rgb-expanded-v1'


class Detector:
    def __init__(self):
        self.model=base.ObjectDetector()
    def detect(self,image,deadline=None):
        _check_deadline(deadline);blob,ratio=base.preprocess(image)
        with self.model._lock:
            self.model._net.setInput(blob);values=np.asarray(self.model._net.forward(self.model._output_name),dtype=np.float32)
        _check_deadline(deadline)
        if values.shape!=(1,8400,85) or not np.isfinite(values).all():raise ValueError('Invalid detector output')
        values=values[0];scores=values[:,4,None]*values[:,5:];class_ids=np.argmax(scores,axis=1);confidence=scores[np.arange(len(values)),class_ids]
        centers=(values[:,:2]+base._GRIDS)*base._STRIDES;sizes=np.exp(np.clip(values[:,2:4],-20,20))*base._STRIDES
        boxes=np.concatenate((centers-sizes/2,sizes),axis=1)/ratio;h,w=image.shape[:2];result=[]
        for label in CLASSES:
            indices=np.flatnonzero((class_ids==label)&(confidence>=.35))
            if not len(indices):continue
            keep=cv2.dnn.NMSBoxes(boxes[indices].tolist(),confidence[indices].tolist(),.35,.5)
            for selected in np.asarray(keep,dtype=int).reshape(-1):
                i=int(indices[selected]);x,y,bw,bh=boxes[i];b=np.clip([x/w,y/h,(x+bw)/w,(y+bh)/h],0,1).tolist()
                if b[2]<=b[0] or b[3]<=b[1]:continue
                result.append({'class_id':label,'class_name':CLASSES[label],'score':float(confidence[i]),'box_xyxy':b})
        return sorted(result,key=lambda x:-x['score'])[:100]


_instance=None
_lock=threading.Lock()
def get_detector():
    global _instance
    with _lock:
        if _instance is None:_instance=Detector()
    return _instance
