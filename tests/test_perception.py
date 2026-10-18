import numpy as np
from service.pipeline.tracker import Tracker,foreground_people
from service.pipeline.detector import Detector


def det(box,kind=0,score=.9):return {'class_id':kind,'class_name':'person' if kind==0 else 'cup','score':score,'box_xyxy':box}


def test_tracks_survive_detection_order_and_short_occlusion():
    t=Tracker();first=t.update([det([.1,.2,.3,.7]),det([.6,.2,.8,.7],score=.8)],0)
    ids={round(r['box_xyxy'][0],1):r['track_id'] for r in first}
    t.update([], .5)
    second=t.update([det([.61,.2,.81,.7],score=.95),det([.11,.2,.31,.7],score=.7)],1)
    assert {round(r['box_xyxy'][0],1):r['track_id'] for r in second}==ids


def test_small_background_people_do_not_inflate_room_occupancy():
    result=foreground_people([det([.1,.3,.3,.9]),det([.5,.05,.6,.15])])
    assert len(result)==1 and result[0]['box_xyxy'][1]==.3


def test_expanded_detector_keeps_cup_and_uses_verified_rgb(monkeypatch):
    from service.pipeline import detector as d
    import threading
    raw=np.zeros((1,8400,85),np.float32)
    raw[0,0,:4]=[10,10,2,2];raw[0,0,4]=.9;raw[0,0,5+41]=.9
    class Net:
        def setInput(self,blob):self.blob=blob;assert blob[0,0,0,0]==255 and blob[0,2,0,0]==0
        def forward(self,name):return raw
    class Base:
        def __init__(self):self._net=Net();self._lock=threading.Lock();self._output_name='out'
    monkeypatch.setattr(d.base,'ObjectDetector',Base)
    image=np.zeros((64,64,3),np.uint8);image[:,:,2]=255
    result=Detector().detect(image)
    assert len(result)==1 and result[0]['class_id']==41
    assert all(0<=v<=1 for v in result[0]['box_xyxy'])


def test_missing_boxes_do_not_prove_absence_or_screen_unlock():
    from service.pipeline.scene import calibrate
    from service.pipeline.facts import timeline
    rows={'frames':[dict(timestamp_sec=0,source_frame_index=0,detections=[],foreground_person_tracks=[])]}
    scene=calibrate(rows);facts=timeline(rows,scene)[0]['facts']
    assert scene['safe'] is None and scene['closed_safe_reference'] is None
    assert facts['staff_present'] is None and facts['cash_on_desk'] is None and facts['laptop_screen_on'] is None
