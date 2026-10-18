"""Optional motion-selected short action inspection; original source evidence only."""
import time
from typing import Literal
import cv2
import numpy as np
from pydantic import BaseModel,ConfigDict,Field,StrictBool,StrictInt
from service.pipeline.vlm import ask

KINDS={'personal_items','food_drink','hand_to_hand_transfer'}

class Action(BaseModel):
    model_config=ConfigDict(extra='forbid')
    kind:Literal['personal_items','food_drink','hand_to_hand_transfer']
    before_frame:StrictInt=Field(ge=0,le=15)
    after_frame:StrictInt=Field(ge=0,le=15)
    before_fact:str=Field(min_length=1,max_length=180)
    after_fact:str=Field(min_length=1,max_length=180)
    support:Literal['clear','uncertain','not_visible']
    prerequisites_visible:StrictBool
    counterexamples_excluded:StrictBool

class ActionSweep(BaseModel):
    model_config=ConfigDict(extra='forbid')
    visibility:Literal['clear','limited','insufficient']
    assessment:Literal['concern','clear_negative','uncertain']
    observations:list[Action]=Field(max_length=3)

def motion_windows(frames,deadline):
    """Rank motion peaks, suppress overlapping8s intervals, retain chronological frames.

    Global brightness changes are removed, not interpreted as action. Selection
    is only a proposal mechanism; motion never establishes any violation.
    """
    if len(frames)<2:return []
    scores=[];previous=None
    for index,frame in enumerate(frames):
        if time.monotonic()>=deadline:raise TimeoutError('Action window selection deadline')
        if index and frame['timestamp_sec']<=frames[index-1]['timestamp_sec']:
            raise ValueError('Source times must strictly increase')
        image=cv2.imdecode(np.frombuffer(frame['image'],np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None:raise ValueError('Invalid source JPEG')
        small=cv2.resize(image,(96,96),interpolation=cv2.INTER_AREA).astype(np.float32)
        if previous is not None:
            difference=small-previous
            difference=np.abs(difference-np.median(difference))
            # Mean of highest10% emphasizes local movement without one-pixel peaks.
            values=difference.ravel();top=max(1,len(values)//10)
            score=float(np.partition(values,len(values)-top)[-top:].mean())
            if score>=2.:scores.append((score,index))
        previous=small
    chosen=[]
    for score,index in sorted(scores,key=lambda x:(-x[0],x[1])):
        center=frames[index]['timestamp_sec'];left=center-4.;right=center+4.
        if any(not(right<start or left>end) for start,end,_ in chosen):continue
        selected=[f for f in frames if left<=f['timestamp_sec']<=right]
        if len(selected)<2:continue
        # Sampling is local only; no duration/class decisions are made here.
        indices=np.linspace(0,len(selected)-1,min(16,len(selected)),dtype=int)
        chosen.append((left,right,[selected[int(i)] for i in indices]))
        if len(chosen)==2:break
    return [entry[2] for entry in sorted(chosen,key=lambda entry:entry[0])]

def action_rescan(frames,deadline,*,requester=None):
    from service.pipeline.auditor import predicate_catalogue
    logs=[];events=[]
    def incomplete(code):
        logs.append(dict(stage='action_window_incomplete',failed_stage='action_windows',error_code=code))
    if time.monotonic()+35>=deadline:
        incomplete('ActionWindowBudget');return events,logs
    try:windows=motion_windows(frames,deadline)
    except (ValueError,TimeoutError) as exc:
        incomplete(type(exc).__name__);return events,logs
    logs.append(dict(stage='action_window_selection',source_times=[[f['timestamp_sec'] for f in w] for w in windows]))
    catalogue='\n'.join(line for line in predicate_catalogue().splitlines() if line.partition(':')[0] in KINDS)
    prompt=('These are chronological original images from one short motion-selected interval, not the entire recording. '
            'observations contains POSITIVE target actions only, never a checklist of all three categories. '
            'Do not add entries describing absence, unchanged empty surfaces or actions that did not occur. '
            'When no target action is visible, return observations=[]; use assessment=clear_negative when clearly observable, otherwise uncertain. '
            'First identify a literal visible BEFORE state and AFTER state of an object-person interaction; then classify only clearly supported actions under the predicates below. '
            'For eating/drinking see actual food/container-to-mouth interaction; for personal items see an unworn coat/personal handbag deposited on the actual cash working surface; '
            'for transfer see the actual object move from one person to another and the bypassed transfer device. Hand proximity alone is insufficient. '
            'Do not infer any missing transaction step, offscreen absence, authorization, ownership, cash storage duty, staff departure or readable session. '
            'Reject worn clothing, operational collection bags, ordinary work documents and legitimate use of a transfer device. '
            'prerequisites_visible=true only when the needed cash-workplace/role/device context is directly visible in THESE images; unknown otherwise. '
            'Use indices in THIS request. before_frame must precede after_frame; choose after_frame as the FIRST clearly visible target action, not its end. '
            'Each Russian before_fact and after_fact must be at most six words, with no legal conclusions. clear_negative means no supported target action in THIS interval only; otherwise uncertain. '+catalogue)
    for selected in windows:
        if time.monotonic()+35>=deadline:
            incomplete('ActionWindowBudget');break
        try:
            answer,meta=ask(selected,prompt,ActionSweep,deadline,requester=requester,max_tokens=512)
        except (OSError,TimeoutError,ValueError,KeyError) as exc:
            incomplete(type(exc).__name__);break
        logs.append(dict(stage='action_window',**meta,answer=answer.model_dump()))
        accepted=0
        for candidate in answer.observations:
            if (candidate.support!='clear' or not candidate.prerequisites_visible or not candidate.counterexamples_excluded
                or not 0<=candidate.before_frame<candidate.after_frame<len(selected)):
                incomplete('ActionWindowUnsupported');continue
            events.append(dict(type_id=candidate.kind,timecode_sec=selected[candidate.after_frame]['timestamp_sec'],
                               evidence_sec=selected[candidate.after_frame]['timestamp_sec'],visible_fact=candidate.after_fact,
                               confidence=.8,origin='motion_selected_action',complete_phase_visible=False))
            accepted+=1
        if answer.visibility!='clear' or (not accepted and not(answer.assessment=='clear_negative' and not answer.observations)) or (accepted and answer.assessment!='concern'):
            incomplete('ActionWindowUnknown')
    return events,logs
