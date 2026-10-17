"""Sequential source-time sampling; no labels, fixed camera zones or audio."""
import math
import cv2
from src.video_analysis import _open_video,_check_deadline


def iter_frames(path,deadline=None,sample_fps=2):
    cap,fps,expected=_open_video(path,deadline);index=0;next_time=0.;last=-1.
    try:
        while cap.grab():
            _check_deadline(deadline)
            if index>=18000:raise ValueError('Decoded frame limit exceeded')
            t=float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
            if not math.isfinite(t) or t<0 or (index and t<=last):t=index/fps
            last=t
            if t+1e-6>=next_time:
                ok,image=cap.retrieve()
                if not ok or image is None:raise ValueError('Source sample decoding failed')
                yield {'timestamp_sec':t,'source_frame_index':index,'image':image}
                next_time=t+1/sample_fps
            index+=1
    finally:cap.release()
    if not index or expected and index<expected-max(2,int(expected*.01)):raise ValueError('Truncated/empty source video')
