"""Conservative per-recording relative geometry; semantic regions may be unknown."""
import numpy as np
from service.pipeline.tracker import iou


def calibrate(observations):
    tracks={};anchors=[];tables=[]
    for row in observations['frames']:
        if row['timestamp_sec']>10:break
        for d in row['detections']:
            if d['class_id'] in [56,63]:anchors.append(d['box_xyxy'])
            if d['class_id']==60:tables.append(d['box_xyxy'])
            if d['track_id'] in row['foreground_person_tracks']:tracks.setdefault(d['track_id'],[]).append(d['box_xyxy'])
    # A cabinet is often mislabeled 'display'. Never reinterpret that as safe,
    # screen lock, or a known closed-door reference.
    anchor=np.median(anchors,axis=0).tolist() if anchors else None
    ranked=[]
    for identifier,boxes in tracks.items():
        box=np.median(boxes,axis=0).tolist();near=iou(box,anchor) if anchor else 0
        ranked.append((len(boxes)*(1+near),identifier,box))
    ranked.sort(reverse=True)
    staff=ranked[0] if ranked and len(tracks[ranked[0][1]])>=4 else None
    return dict(coordinates='normalized_xyxy',cashier_track=staff[1] if staff else None,
                cashier_zone=staff[2] if staff else None,work_anchor=anchor,
                desk_candidates=tables,safe=None,door=None,closed_safe_reference=None,
                role_status='geometric_proposal_only',glass_separation_verified=False)
