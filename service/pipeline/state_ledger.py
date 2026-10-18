"""Opt-in literal per-frame storage/door ledger; original OpenState unchanged."""
import time
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,StrictInt
from service.pipeline.vlm import ask
from service.pipeline.spatial import GroundedBox,bounded_box,crop_pair,state_samples

class FrameState(BaseModel):
    model_config=ConfigDict(extra='forbid')
    frame:StrictInt=Field(ge=0,le=63)
    state:Literal['open','closed','unknown']
    operator:Literal['accessing','passing','clear','unknown']

class StateLedger(BaseModel):
    model_config=ConfigDict(extra='forbid')
    frames:list[FrameState]=Field(min_length=1,max_length=64)

def storage_counterexample(candidate,frames,onset,evidence,deadline,*,requester=None):
    """Optional veto authority only; retain proposals on unresolved inspection.

    Reuses exact ledger requests. A positive ledger does not replace the original
    event/time, and failed auxiliary checks cannot make primary evidence fail.
    """
    if candidate.kind!='safe_left_open':return False,[]
    try:
        event,logs=verify_open_state(candidate,frames,onset,evidence,deadline,requester=requester)
    except (OSError,TimeoutError,ValueError,KeyError) as exc:
        return False,[dict(stage='storage_veto_incomplete',kind=candidate.kind,error_code=type(exc).__name__)]
    unresolved=any(row.get('stage')=='audit_incomplete' for row in logs)
    veto=(event is None and not unresolved and
          any(row.get('stage')=='spatial_ledger_decision' and row.get('decision')=='counterexample' for row in logs))
    logs=[dict(row,stage='storage_veto_incomplete') if row.get('stage')=='audit_incomplete' else row for row in logs]
    logs.append(dict(stage='storage_veto_decision',kind=candidate.kind,veto=veto))
    return veto,logs

def ledger_interval(answer,count,kind):
    """An observed open access -> open after clearance; never bridge unknowns.

    No dwell threshold or inferred absence. A later closure/occlusion cannot
    invalidate an already observed transition. Indices must cover every input.
    """
    rows=answer.frames
    if [row.frame for row in rows]!=list(range(count)):
        return 'unknown',None
    active={'passing'} if kind=='door_left_open' else {'accessing'}
    for previous,current in zip(rows,rows[1:]):
        if (previous.state=='open' and previous.operator in active
                and current.state=='open' and current.operator=='clear'):
            return 'event',current.frame
    if any(row.state=='unknown' or row.operator=='unknown' for row in rows):
        return 'unknown',None
    # All sampled observations are explicit; no qualifying transition was seen.
    # An already-open object without observed access is unresolved, not clean.
    if any(row.state=='open' and row.operator=='clear' for row in rows):
        return 'unknown',None
    return 'counterexample',None

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
        question=('Describe literal states for EVERY chronological image; return one row per frame in exact index order. '
                  'LEFT is the full scene; RIGHT is a padded detail of the same grounded object at the same source time. '
                  'state=open requires an actual aperture/interior or visibly displaced door panel; closed requires a visibly closed panel. '
                  'A seam, handle or dark rectangle is insufficient. Use unknown for occlusion, object mismatch or ambiguous detail. '
                  'operator=accessing means actual ongoing handling of this storage object; passing means a person is still traversing this doorway. '
                  'operator=clear means the SAME previously observed operator is visibly fully clear of this doorway or has visibly moved away from storage and is no longer handling it. '
                  'Use anonymous continuous position/movement only; a different person, unseen operator, pause at the object or uncertain continuity means unknown. '
                  'Do not infer departure from an empty image. Report physical states only, without deciding whether conduct is unsafe, normal, authorized or a violation.')
        answer,meta=ask(cropped,question,StateLedger,deadline,requester=requester,max_tokens=768)
        logs.append(dict(stage='spatial_state_ledger',kind=candidate.kind,grounded_box_1000=list(box),**meta,answer=answer.model_dump()))
    except (OSError,TimeoutError,ValueError,KeyError) as exc:return incomplete(type(exc).__name__)
    decision,index=ledger_interval(answer,len(selected),candidate.kind)
    logs.append(dict(stage='spatial_ledger_decision',kind=candidate.kind,decision=decision,onset_frame=index))
    if decision=='unknown':return incomplete('SpatialLedgerUnresolved')
    if decision=='counterexample':return None,logs
    fact=('После видимого прохода человек находится вне дверного проёма, а дверь остаётся открытой.'
          if candidate.kind=='door_left_open' else
          'После видимой работы с хранилищем оператор отошёл от него, а дверца остаётся открытой.')
    return dict(type_id=candidate.kind,timecode_sec=selected[index]['timestamp_sec'],evidence_sec=selected[index]['timestamp_sec'],
                visible_fact=fact,confidence=.8,origin='spatially_confirmed_state_ledger',complete_phase_visible=False),logs
