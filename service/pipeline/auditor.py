"""Chronological physical observations plus independent focused confirmation."""
import json,time
from typing import Literal
import numpy as np
from pydantic import BaseModel,ConfigDict,Field,StrictBool,StrictInt
from service.pipeline.vlm import ask
from src.video_semantic import TYPE_PREDICATES

Kind=Literal['cash_left_out','door_left_open','unauthorized_access','hand_to_hand_transfer','safe_left_open','keys_left_in_safe','workstation_unlocked','counting_without_client','no_count_check','no_id_check','no_receipt','personal_items','food_drink','personal_activity']
OMISSIONS={'no_count_check','no_id_check','no_receipt'}
TINY_OR_AMBIGUOUS={'keys_left_in_safe','workstation_unlocked','personal_activity'}


class UnusableVisibilityError(RuntimeError):pass


class Candidate(BaseModel):
    model_config=ConfigDict(extra='forbid')
    kind:Kind
    onset_frame:StrictInt=Field(ge=0,le=15)
    evidence_frame:StrictInt=Field(ge=0,le=15)
    support:Literal['clear','uncertain','not_visible']
    counterexamples_excluded:StrictBool
    complete_phase_visible:StrictBool
    visible_fact:str=Field(min_length=1,max_length=280)


class Sweep(BaseModel):
    model_config=ConfigDict(extra='forbid')
    visibility:Literal['clear','limited','insufficient']
    observations:list[Candidate]=Field(max_length=10)


class Confirmation(BaseModel):
    model_config=ConfigDict(extra='forbid')
    support:Literal['clear','counterexample','uncertain','not_visible']
    onset_frame:StrictInt=Field(ge=0,le=15)
    evidence_frame:StrictInt=Field(ge=0,le=15)
    complete_phase_visible:StrictBool
    visible_fact:str=Field(min_length=1,max_length=280)


def choose(frames,count=16):
    indices=np.linspace(0,len(frames)-1,min(count,len(frames)),dtype=int)
    return [frames[int(i)] for i in indices]


def confirmed_observations(frames,deadline,*,requester=None):
    overview=choose(frames);log=[];observations=[]
    prompt=('Find clearly visible physical concerns across these chronological cash-workplace frames. '
            'Use only the catalogue predicates below. Return onset_frame and evidence_frame as indices in THIS request. '
            'The onset is when the problematic conduct starts, not when the room finally becomes empty. '
            'Ordinary customer service, operational collection bags, staff working at a safe and people behind glass are counterexamples. '
            'No candidate for unclear conduct. Do not invent authority, personal ownership, readable computer session or tiny keys. '
            'Explain literal visible_fact in Russian. No legal conclusions or quotations. '+TYPE_PREDICATES)
    sweep,meta=ask(overview,prompt,Sweep,deadline,requester=requester,max_tokens=1024);log.append(dict(stage='sweep',**meta,answer=sweep.model_dump()))
    if sweep.visibility=='insufficient':raise UnusableVisibilityError('Auditor could not observe a usable scene')
    for candidate in sweep.observations:
        if candidate.support!='clear' or not candidate.counterexamples_excluded:continue
        if max(candidate.onset_frame,candidate.evidence_frame)>=len(overview):continue
        if candidate.kind in OMISSIONS:
            # The complete phase must be densely covered, not just claimed so
            # by a language model looking at sparse stills.
            if not candidate.complete_phase_visible or len(frames)>16:continue
        onset=overview[candidate.onset_frame]['timestamp_sec'];evidence=overview[candidate.evidence_frame]['timestamp_sec']
        earlier=overview[max(0,min(candidate.onset_frame,candidate.evidence_frame)-1)]['timestamp_sec']
        later=overview[min(len(overview)-1,max(candidate.onset_frame,candidate.evidence_frame)+1)]['timestamp_sec']
        left=max(0,min(min(onset,evidence)-3,earlier-1));right=max(max(onset,evidence)+3,later+1)
        focused=choose([f for f in frames if left<=f['timestamp_sec']<=right])
        if len(focused)<2:continue
        question=('Independently check this possible observation; it may be wrong. Return clear only when actual objects, movement and prerequisites are visible. '
                  'Reject normal work, background/reflected people, ongoing safe access, unidentified bags, lit-screen-as-unlocked, hand proximity-as-transfer. '
                  'A cabinet is not a laptop. Authorization/ownership cannot be identified by appearance. '
                  'Index onset/evidence in THESE focused frames; explain visible_fact in Russian. '
                  'Check physical predicate: '+candidate.kind+'. '+TYPE_PREDICATES)
        if time.monotonic()+11>=deadline:break
        checked,meta=ask(focused,question,Confirmation,deadline,requester=requester,max_tokens=512);log.append(dict(stage='focused',kind=candidate.kind,**meta,answer=checked.model_dump()))
        if checked.support!='clear' or max(checked.onset_frame,checked.evidence_frame)>=len(focused):continue
        if candidate.kind in OMISSIONS and not checked.complete_phase_visible:continue
        if candidate.kind in TINY_OR_AMBIGUOUS:
            if time.monotonic()+11>=deadline:continue
            # A separately worded temporal subsample must corroborate tiny
            # states. Keep actual source timestamps and chronological order.
            second_frames=focused[1::2]
            if len(second_frames)<2:continue
            second,meta=ask(second_frames,question+' This is a different temporal subsample. Verify direct detail; uncertain if not readable.',Confirmation,deadline,requester=requester,max_tokens=512)
            log.append(dict(stage='second_confirmation',kind=candidate.kind,**meta,answer=second.model_dump()))
            if second.support!='clear':continue
        observations.append(dict(type_id=candidate.kind,timecode_sec=focused[checked.onset_frame]['timestamp_sec'],
                                 evidence_sec=focused[checked.evidence_frame]['timestamp_sec'],visible_fact=checked.visible_fact,
                                 confidence=.8,origin='confirmed_self_hosted_observation',complete_phase_visible=checked.complete_phase_visible))
    return observations,log
