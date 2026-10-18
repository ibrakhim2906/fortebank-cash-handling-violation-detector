"""Opt-in dynamically grounded storage/door state verification.

Crops are inspection aids only: returned event times point to original source
frames, and API evidence generation continues to use the unchanged originals.
"""
import math,time
from typing import Literal
import cv2
import numpy as np
from pydantic import BaseModel,ConfigDict,Field,StrictBool,StrictInt
from service.pipeline.vlm import ask,frame_limit

OPEN_STATE={'safe_left_open','door_left_open'}

class GroundedBox(BaseModel):
    model_config=ConfigDict(extra='forbid')
    located:StrictBool
    object_kind:Literal['safe','lockable_storage','door','other','unknown']
    x0:StrictInt=Field(ge=0,le=1000)
    y0:StrictInt=Field(ge=0,le=1000)
    x1:StrictInt=Field(ge=0,le=1000)
    y1:StrictInt=Field(ge=0,le=1000)
    visible_fact:str=Field(min_length=1,max_length=200)

class OpenState(BaseModel):
    model_config=ConfigDict(extra='forbid')
    support:Literal['clear','counterexample','uncertain','not_visible']
    same_object_visible:StrictBool
    actual_opening_visible:StrictBool
    operator_departure_visible:StrictBool
    object_access_finished_visible:StrictBool
    ongoing_access:StrictBool
    unsafe_interval_visible:StrictBool
    normal_access_closure:StrictBool
    final_state:Literal['open','closed','uncertain','not_visible']
    onset_frame:StrictInt=Field(ge=0,le=63)
    evidence_frame:StrictInt=Field(ge=0,le=63)
    visible_fact:str=Field(min_length=1,max_length=280)


def bounded_box(answer,kind):
    """Reject unknown/tiny/full-image boxes rather than guessing camera ROIs."""
    expected={'door'} if kind=='door_left_open' else {'safe','lockable_storage'}
    if not answer.located or answer.object_kind not in expected:return None
    x0,y0,x1,y1=answer.x0,answer.y0,answer.x1,answer.y1
    width=x1-x0;height=y1-y0
    if width<40 or height<40 or not .004<=width*height/1_000_000<=.85:return None
    return x0,y0,x1,y1


def crop_pair(frame,box,padding=.35):
    """Same normalized object region each time; context left, enlarged ROI right."""
    source=cv2.imdecode(np.frombuffer(frame.get('inspection_image',frame['image']),dtype=np.uint8),cv2.IMREAD_COLOR)
    if source is None:raise ValueError('Invalid source JPEG')
    h,w=source.shape[:2];x0,y0,x1,y1=box
    if min((x1-x0)*w/1000,(y1-y0)*h/1000)<12:raise ValueError('Grounded object is too small in actual pixels')
    dx=(x1-x0)*padding;dy=(y1-y0)*padding
    left=max(0,math.floor((x0-dx)*w/1000));right=min(w,math.ceil((x1+dx)*w/1000))
    top=max(0,math.floor((y0-dy)*h/1000));bottom=min(h,math.ceil((y1+dy)*h/1000))
    roi=source[top:bottom,left:right]
    if roi.size==0:raise ValueError('Empty grounded crop')
    height=512;context_width=max(1,round(w*height/h));crop_width=max(1,round(roi.shape[1]*height/roi.shape[0]))
    # Keep original scene geometry and ROI geometry. Cap very wide ROI canvases.
    context=cv2.resize(source,(context_width,height),interpolation=cv2.INTER_AREA)
    detail=cv2.resize(roi,(crop_width,height),interpolation=cv2.INTER_CUBIC)
    composite=np.concatenate([context,detail],axis=1)
    if composite.shape[1]>1600:
        composite=cv2.resize(composite,(1600,round(1600*height/composite.shape[1])),interpolation=cv2.INTER_AREA)
    ok,jpeg=cv2.imencode('.jpg',composite,[cv2.IMWRITE_JPEG_QUALITY,90])
    if not ok:raise ValueError('Grounded crop encoding failed')
    return dict(frame,image=jpeg.tobytes())


def state_samples(frames,onset,evidence):
    """Uniform whole-clip context plus exact proposal neighborhoods and final frame."""
    count=min(frame_limit(),16,len(frames));indices=set(int(i) for i in np.linspace(0,len(frames)-1,count))
    for moment in [onset-2,onset,onset+2,evidence]:
        indices.add(min(range(len(frames)),key=lambda i:abs(frames[i]['timestamp_sec']-moment)))
    # Keep proposal/final anchors when capacity is the default16.
    anchors={0,len(frames)-1,*[min(range(len(frames)),key=lambda i:abs(frames[i]['timestamp_sec']-t)) for t in [onset-2,onset,onset+2,evidence]]}
    if len(indices)>frame_limit():
        extras=sorted(indices-anchors);slots=frame_limit()-len(anchors)
        indices=anchors|{extras[int(i)] for i in np.linspace(0,len(extras)-1,slots,dtype=int)} if slots else anchors
    return [frames[i] for i in sorted(indices)]


def verify_open_state(candidate,frames,onset,evidence,deadline,*,requester=None):
    logs=[]
    def incomplete(reason):
        logs.append(dict(stage='audit_incomplete',kind=candidate.kind,error_code=reason));return None,logs
    if time.monotonic()+35>=deadline:return incomplete('SpatialBudget')
    focus=min(frames,key=lambda f:abs(f['timestamp_sec']-evidence))
    target='physical safe or lockable cash-storage cabinet' if candidate.kind=='safe_left_open' else 'physical room-entry door and its frame'
    prompt=('Locate the '+target+' in this original source image. The proposed concern may be false. '
            'Return one tight bounding rectangle for the complete actual object, coordinates normalized0..1000 across this image. '
            'Locate the object whether open or closed; do not infer its state here. A laptop, dark seam, reflection or arbitrary rectangular background is not storage. '
            'Return located=false and object_kind=unknown if the object cannot be distinguished. Do not guess a camera region. Explain visible_fact briefly in Russian.')
    try:
        box_answer,meta=ask([focus],prompt,GroundedBox,deadline,requester=requester,max_tokens=256)
        logs.append(dict(stage='spatial_grounding',kind=candidate.kind,**meta,answer=box_answer.model_dump()))
        box=bounded_box(box_answer,candidate.kind)
        if box is None:return incomplete('SpatialObjectUnknown')
        if time.monotonic()+15>=deadline:return incomplete('SpatialStateBudget')
        selected=state_samples(frames,onset,evidence);cropped=[crop_pair(f,box) for f in selected]
        question=('Independently check this possibly wrong '+candidate.kind+' proposal. Each chronological image has the original full scene LEFT and an enlarged padded crop of the SAME grounded object RIGHT, at the same actual source time. '
                  'Inspect the first image, action/departure transition, subsequent images and FINAL image. '
                  'An open state requires actual visible aperture/interior or displaced door panel: a handle, closed panel, dark seam, laptop screen or cabinet outline is insufficient. '
                  'Confirm the same object throughout; camera/object shifts or hidden detail are uncertain. '
                  'Reject ordinary passage and staff actively opening/using storage. Set normal_access_closure=true when opening and closing are one ordinary attended access sequence. '
                  'Set unsafe_interval_visible=true only when a real interval directly shows the object open AFTER the operator departs or the object access ends, with no ongoing attended access. ongoing_access refers to the alleged unsafe interval itself, not ordinary access in some earlier or later frames. Return clear only for such an actual unsafe interval. '
                  'Inspect the final image and report its actual final_state, including uncertain or not_visible when later occluded. Later closure or later occlusion does not erase an earlier directly observed unsafe interval; a single ordinary attended open-close sequence without an unsafe interval is a counterexample. '
                  'Choose onset at actual departure/end of access, not first object appearance; choose evidence where unsafe open state is directly visible. '
                  'Use indices in THESE inspection images, explain visible_fact in Russian, and never infer identity, authorization, lock/key status or legal fault.')
        answer,meta=ask(cropped,question,OpenState,deadline,requester=requester,max_tokens=512)
        logs.append(dict(stage='spatial_open_state',kind=candidate.kind,grounded_box_1000=list(box),**meta,answer=answer.model_dump()))
    except (OSError,TimeoutError,ValueError,KeyError) as exc:return incomplete(type(exc).__name__)
    if answer.support in ['uncertain','not_visible']:return incomplete('SpatialStateUnknown')
    if answer.support!='clear' or answer.ongoing_access:return None,logs
    if not answer.unsafe_interval_visible:
        if answer.normal_access_closure:return None,logs
        return incomplete('SpatialUnsafeIntervalUnproven')
    if not answer.same_object_visible or not answer.actual_opening_visible:return incomplete('SpatialOpeningUnproven')
    if not (answer.operator_departure_visible or answer.object_access_finished_visible):return incomplete('SpatialDepartureUnproven')
    if max(answer.onset_frame,answer.evidence_frame)>=len(selected):return incomplete('SpatialIndexOutOfRange')
    return dict(type_id=candidate.kind,timecode_sec=selected[answer.onset_frame]['timestamp_sec'],evidence_sec=selected[answer.evidence_frame]['timestamp_sec'],
                visible_fact=answer.visible_fact,confidence=.8,origin='spatially_confirmed_self_hosted_observation',complete_phase_visible=False),logs
