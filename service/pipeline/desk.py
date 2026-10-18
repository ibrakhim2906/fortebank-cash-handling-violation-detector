"""Opt-in second look at grounded worktop/staff details after no accepted concern."""
import time
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,StrictBool,StrictInt
from service.pipeline.vlm import ask
from service.pipeline.spatial import crop_pair

KINDS={'personal_items','food_drink','counting_without_client','cash_left_out','hand_to_hand_transfer'}


class WorkRegion(BaseModel):
    model_config=ConfigDict(extra='forbid')
    located:StrictBool
    consistent_across_views:StrictBool
    x0:StrictInt=Field(ge=0,le=1000)
    y0:StrictInt=Field(ge=0,le=1000)
    x1:StrictInt=Field(ge=0,le=1000)
    y1:StrictInt=Field(ge=0,le=1000)


class DeskCandidate(BaseModel):
    model_config=ConfigDict(extra='forbid')
    kind:Literal['personal_items','food_drink','counting_without_client','cash_left_out','hand_to_hand_transfer']
    onset_frame:StrictInt=Field(ge=0,le=15)
    evidence_frame:StrictInt=Field(ge=0,le=15)
    support:Literal['clear','uncertain','not_visible']
    counterexamples_excluded:StrictBool
    visible_fact:str=Field(min_length=1,max_length=280)


class DeskSweep(BaseModel):
    model_config=ConfigDict(extra='forbid')
    visibility:Literal['clear','limited','insufficient']
    assessment:Literal['concern','clear_negative','uncertain']
    observations:list[DeskCandidate]=Field(max_length=10)


def work_box(answer):
    if not answer.located or not answer.consistent_across_views:return None
    width=answer.x1-answer.x0;height=answer.y1-answer.y0
    if width<80 or height<80 or not .01<=width*height/1_000_000<=.95:return None
    return answer.x0,answer.y0,answer.x1,answer.y1


def desk_rescan(frames,deadline,*,requester=None):
    from service.pipeline.auditor import choose,predicate_catalogue
    logs=[]
    def incomplete(code):
        logs.append(dict(stage='audit_incomplete',error_code=code));return [],logs
    if len(frames)<2 or time.monotonic()+35>=deadline:return incomplete('DeskRescanBudget')
    # Inspection originals remain auxiliary; never alter coarse/API source bytes.
    contexts=[dict(frame,image=frame.get('inspection_image',frame['image'])) for frame in choose(frames,3)]
    grounding=('Locate one shared region covering the actual cash working surface and the operating staff member\'s upper body/hands across these first/middle/last source images. '
               'Use visible work position and physical equipment, never identity or clothing assumptions. Include worktop objects and hand-to-mouth movement where visible. '
               'Return a common rectangle in normalized0..1000 original-image coordinates, with consistent_across_views=false if camera movement or layout prevents one shared region. '
               'Do not invent an offscreen desk/staff region; located=false when it cannot be grounded. No conduct or violation judgment.')
    try:
        region,meta=ask(contexts,grounding,WorkRegion,deadline,requester=requester,max_tokens=256)
        logs.append(dict(stage='desk_grounding',**meta,answer=region.model_dump()))
        box=work_box(region)
        if box is None:return incomplete('DeskRegionUnknown')
        if time.monotonic()+15>=deadline:return incomplete('DeskRescanStateBudget')
        selected=choose(frames,16);cropped=[crop_pair(frame,box) for frame in selected]
        catalogue='\n'.join(line for line in predicate_catalogue().splitlines() if line.partition(':')[0] in KINDS)
        question=('Inspect chronological worktop/staff details. Each image shows original full context LEFT and a padded crop RIGHT at the same source time. '
                  'Report only clearly visible concerns under the five predicates below. Use indices in THIS request for onset/evidence. '
                  'Onset is the actual problematic action or end-of-work/departure transition, not first object appearance. '
                  'Establish worker versus customer from visible work actions; a customer leaving does not establish worker departure. '
                  'Reject active legitimate cash handling, operational bags, worn clothing, work documents, background people and hand proximity without actual transfer. '
                  'A crop is not evidence that a person or step outside it is absent: use the full context. '
                  'No omission, identity, authority, ownership, tiny-key or computer-session inference. Unknown stays unknown. '
                  'assessment=clear_negative only when visibility is clear and the observed scene clearly supports no concern under these predicates; otherwise uncertain. '
                  'Return all supported observations and literal Russian visible_fact, without legal conclusions. '+catalogue)
        answer,meta=ask(cropped,question,DeskSweep,deadline,requester=requester,max_tokens=1024)
        logs.append(dict(stage='desk_rescan',grounded_box_1000=list(box),**meta,answer=answer.model_dump()))
    except (OSError,TimeoutError,ValueError,KeyError) as exc:return incomplete(type(exc).__name__)
    events=[];uncertain=False
    for candidate in answer.observations:
        if candidate.support!='clear' or not candidate.counterexamples_excluded:
            uncertain=True;continue
        if max(candidate.onset_frame,candidate.evidence_frame)>=len(selected):
            uncertain=True;continue
        events.append(dict(type_id=candidate.kind,timecode_sec=selected[candidate.onset_frame]['timestamp_sec'],
                           evidence_sec=selected[candidate.evidence_frame]['timestamp_sec'],visible_fact=candidate.visible_fact,
                           confidence=.8,origin='grounded_worktop_rescan',complete_phase_visible=False))
    if events:
        if uncertain or answer.visibility!='clear' or answer.assessment!='concern':
            logs.append(dict(stage='audit_incomplete',error_code='DeskPartialEvidence'))
        return events,logs
    if answer.visibility=='clear' and answer.assessment=='clear_negative' and not answer.observations:return [],logs
    return incomplete('DeskRescanUnknown')
