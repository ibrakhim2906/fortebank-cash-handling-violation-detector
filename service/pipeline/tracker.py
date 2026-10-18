"""Within-video anonymous geometric tracks, no faces or persistent identities."""
import math


def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return overlap/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)


class Tracker:
    def __init__(self):self.tracks={};self.next_id=0
    def update(self,detections,t):
        self.tracks={i:r for i,r in self.tracks.items() if t-r['last_seen']<=1.5};available=set(self.tracks);result=[]
        for d in sorted(detections,key=lambda r:-r['score']):
            match=None;best=.25
            for i in sorted(available):
                r=self.tracks[i]
                if r['class_id']!=d['class_id']:continue
                score=iou(r['box_xyxy'],d['box_xyxy'])
                if score>best:best=score;match=i
            if match is None:match=self.next_id;self.next_id+=1
            else:available.remove(match)
            r={**d,'track_id':match,'last_seen':t};self.tracks[match]=r;result.append(r)
        return result


def foreground_people(detections):
    people=[d for d in detections if d['class_id']==0]
    if not people:return []
    largest=max(d['box_xyxy'][3]-d['box_xyxy'][1] for d in people)
    return [d for d in people if d['box_xyxy'][3]-d['box_xyxy'][1]>=max(.14,largest*.55)]
