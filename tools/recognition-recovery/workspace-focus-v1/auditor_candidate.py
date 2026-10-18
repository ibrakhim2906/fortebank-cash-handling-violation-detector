"""Chronological physical observations plus independent focused confirmation."""
import json,os,time
from typing import Literal
import numpy as np
from pydantic import BaseModel,ConfigDict,Field,StrictBool,StrictInt
from service.pipeline.vlm import ask,frame_limit
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


class Onset(BaseModel):
    model_config=ConfigDict(extra='forbid')
    located:StrictBool
    onset_frame:StrictInt=Field(ge=0,le=15)
    evidence_frame:StrictInt=Field(ge=0,le=15)
    visible_transition:str=Field(min_length=1,max_length=280)


class WideCandidate(Candidate):
    onset_frame:StrictInt=Field(ge=0,le=63)
    evidence_frame:StrictInt=Field(ge=0,le=63)


class WideSweep(Sweep):
    observations:list[WideCandidate]=Field(max_length=10)


class WideConfirmation(Confirmation):
    onset_frame:StrictInt=Field(ge=0,le=63)
    evidence_frame:StrictInt=Field(ge=0,le=63)


class WideOnset(Onset):
    onset_frame:StrictInt=Field(ge=0,le=63)
    evidence_frame:StrictInt=Field(ge=0,le=63)


def choose(frames,count=16):
    indices=np.linspace(0,len(frames)-1,min(count,len(frames)),dtype=int)
    return [frames[int(i)] for i in indices]


def predicate_catalogue():
    mode=os.environ.get('AUDITOR_WORKFLOW','original')
    if mode not in ['original','storage_context']:raise ValueError('Unknown workflow context')
    if mode=='original':return TYPE_PREDICATES
    definition=('cash_left_out: identifiable banknotes, or a collection bag whose cash contents were actually visible earlier in this recording, remain outside designated lockable storage after the visible cash-handling phase has ended or after the established cashier leaves. '
                'Distinguish the cashier using actual cash-equipment/storage work from the client being served; client departure alone is NOT cashier departure. '
                'Previously accepted cash put aside while work continues elsewhere may be a storage concern even if the cashier remains seated. '
                'Active counting/preparation/handover, temporary client cash during service and unidentified bags are counterexamples; mere cash presence is insufficient.')
    return ''.join(definition+'\n' if line.startswith('cash_left_out:') else line for line in TYPE_PREDICATES.splitlines(keepends=True))


def refine_onset(event,frames,deadline,*,requester=None):
    """Locate the duty/state transition globally, then inspect its interval.

    This tests timing independently of the proposal's video-level decision.
    A failed/uncertain timing estimate retains the source-backed proposal.
    """
    overview=choose(frames,frame_limit());log=[];current=dict(event)
    predicate=next(line for line in predicate_catalogue().splitlines() if line.startswith(event['type_id']+':'))
    question=('Locate the transition for this proposed physical concern in chronological source images. '
              'The onset is when conduct becomes problematic, not when an object or open door first appears. '
              'For a departure concern, locate actual departure leaving the unsafe state; earlier active passage, cash handling or safe access is not its onset. '
              'For cash storage, locate when the relevant handling phase ends or supervision ceases, rather than first cash appearance. '
              'For entry, transfer or drinking, locate the actual action. Return located=false if the transition cannot be localized. '
              'Use indices in THIS request, and describe only the visible transition in Russian. '+predicate)
    for stage,samples in [('global_onset',overview),('dense_onset',None)]:
        if samples is None:
            center=min(range(len(overview)),key=lambda i:abs(overview[i]['timestamp_sec']-current['timecode_sec']))
            left=overview[max(0,center-1)]['timestamp_sec'];right=overview[min(len(overview)-1,center+1)]['timestamp_sec']
            samples=choose([f for f in frames if left<=f['timestamp_sec']<=right])
        if len(samples)<2 or time.monotonic()+25>=deadline:
            log.append(dict(stage='timing_refinement_skipped',kind=event['type_id']));break
        try:
            answer,meta=ask(samples,question,WideOnset if frame_limit()==64 else Onset,deadline,requester=requester,max_tokens=256)
        except (OSError,TimeoutError,ValueError,KeyError) as exc:
            log.append(dict(stage='timing_refinement_skipped',kind=event['type_id'],error_code=type(exc).__name__));break
        log.append(dict(stage=stage,kind=event['type_id'],**meta,answer=answer.model_dump()))
        if not answer.located or max(answer.onset_frame,answer.evidence_frame)>=len(samples):break
        current.update(timecode_sec=samples[answer.onset_frame]['timestamp_sec'],evidence_sec=samples[answer.evidence_frame]['timestamp_sec'],visible_fact=answer.visible_transition)
    return current,log


def confirmed_observations(frames,deadline,*,requester=None,overview_count=None,allow_omissions=True):
    action_windows=os.environ.get('AUDITOR_ACTION_WINDOWS','off')
    if action_windows not in ['off','on']:raise ValueError('Unknown action window mode')
    desk=os.environ.get('AUDITOR_DESK_RESCAN','off')
    if desk not in ['off','on']:raise ValueError('Unknown desk rescan mode')
    spatial=os.environ.get('AUDITOR_SPATIAL','off')
    if spatial not in ['off','open_state','state_ledger','storage_veto']:raise ValueError('Unknown spatial verification mode')
    policy=os.environ.get('AUDITOR_CONFIRMATION','strict')
    if policy not in ['strict','proposal','proposal_refine']:raise ValueError('Unknown confirmation policy')
    capacity=frame_limit();overview=choose(frames,overview_count or capacity);log=[];observations=[];catalogue=predicate_catalogue()
    sweep_schema=WideSweep if capacity==64 else Sweep;confirmation_schema=WideConfirmation if capacity==64 else Confirmation
    bag_exception='ordinary handling of collection bags' if os.environ.get('AUDITOR_WORKFLOW')=='storage_context' else 'operational collection bags'
    prompt=('Find clearly visible physical concerns across these chronological cash-workplace frames. '
            'Use only the catalogue predicates below. Return onset_frame and evidence_frame as indices in THIS request. '
            'The onset is when the problematic conduct starts, not when the room finally becomes empty. '
            'Ordinary customer service, '+bag_exception+', staff working at a safe and people behind glass are counterexamples. '
            'No candidate for unclear conduct. Do not invent authority, personal ownership, readable computer session or tiny keys. '
            'Explain literal visible_fact in Russian. No legal conclusions or quotations. '+catalogue)
    sweep,meta=ask(overview,prompt,sweep_schema,deadline,requester=requester,max_tokens=1024);log.append(dict(stage='sweep',**meta,answer=sweep.model_dump()))
    if sweep.visibility=='insufficient':raise UnusableVisibilityError('Auditor could not observe a usable scene')
    for candidate in sweep.observations:
        if candidate.support!='clear' or not candidate.counterexamples_excluded:continue
        if max(candidate.onset_frame,candidate.evidence_frame)>=len(overview):continue
        if candidate.kind in OMISSIONS:
            # The complete phase must be densely covered, not just claimed so
            # by a language model looking at sparse stills.
            gaps=any(b['timestamp_sec']-a['timestamp_sec']>.600001 for a,b in zip(overview,overview[1:]))
            if not allow_omissions or not candidate.complete_phase_visible or len(frames)>len(overview) or gaps:continue
        onset=overview[candidate.onset_frame]['timestamp_sec'];evidence=overview[candidate.evidence_frame]['timestamp_sec']
        if spatial=='storage_veto' and candidate.kind=='safe_left_open':
            from service.pipeline.state_ledger import storage_counterexample
            veto,details=storage_counterexample(candidate,frames,onset,evidence,deadline,requester=requester)
            log.extend(details)
            if veto:continue
        if spatial in ['open_state','state_ledger'] and candidate.kind in {'safe_left_open','door_left_open'}:
            if spatial=='state_ledger':
                from service.pipeline.state_ledger import verify_open_state
            else:
                from service.pipeline.spatial import verify_open_state
            event,details=verify_open_state(candidate,frames,onset,evidence,deadline,requester=requester)
            log.extend(details)
            if event is not None:observations.append(event)
            continue
        if policy in ['proposal','proposal_refine'] and candidate.kind not in OMISSIONS|TINY_OR_AMBIGUOUS:
            event=dict(type_id=candidate.kind,timecode_sec=onset,evidence_sec=evidence,
                       visible_fact=candidate.visible_fact,confidence=.8,
                       origin='single_pass_self_hosted_observation',complete_phase_visible=candidate.complete_phase_visible)
            if policy=='proposal_refine':
                event,details=refine_onset(event,frames,deadline,requester=requester);log.extend(details)
            observations.append(event)
            continue
        earlier=overview[max(0,min(candidate.onset_frame,candidate.evidence_frame)-1)]['timestamp_sec']
        later=overview[min(len(overview)-1,max(candidate.onset_frame,candidate.evidence_frame)+1)]['timestamp_sec']
        left=max(0,min(min(onset,evidence)-3,earlier-1));right=max(max(onset,evidence)+3,later+1)
        focused=overview if candidate.kind in OMISSIONS else choose([f for f in frames if left<=f['timestamp_sec']<=right],capacity)
        if len(focused)<2:continue
        question=('Independently check this possible observation; it may be wrong. Return clear only when actual objects, movement and prerequisites are visible. '
                  'Reject normal work, background/reflected people, ongoing safe access, unidentified bags, lit-screen-as-unlocked, hand proximity-as-transfer. '
                  'A cabinet is not a laptop. Authorization/ownership cannot be identified by appearance. '
                  'Index onset/evidence in THESE focused frames; explain visible_fact in Russian. '
                  'Check physical predicate: '+candidate.kind+'. '+catalogue)
        if time.monotonic()+11>=deadline:
            log.append(dict(stage='audit_incomplete',error_code='BudgetLimit'));continue
        try:
            checked,meta=ask(focused,question,confirmation_schema,deadline,requester=requester,max_tokens=512)
        except (OSError,TimeoutError,ValueError,KeyError) as exc:
            log.append(dict(stage='audit_incomplete',kind=candidate.kind,error_code=type(exc).__name__))
            continue
        log.append(dict(stage='focused',kind=candidate.kind,**meta,answer=checked.model_dump()))
        if checked.support!='clear' or max(checked.onset_frame,checked.evidence_frame)>=len(focused):continue
        if candidate.kind in OMISSIONS and not checked.complete_phase_visible:continue
        if candidate.kind in TINY_OR_AMBIGUOUS:
            if time.monotonic()+11>=deadline:
                log.append(dict(stage='audit_incomplete',error_code='TinyDetailBudget'));continue
            # A separately worded temporal subsample must corroborate tiny
            # states. Keep actual source timestamps and chronological order.
            second_frames=focused[1::2]
            if len(second_frames)<2:continue
            try:
                second,meta=ask(second_frames,question+' This is a different temporal subsample. Verify direct detail; uncertain if not readable.',confirmation_schema,deadline,requester=requester,max_tokens=512)
            except (OSError,TimeoutError,ValueError,KeyError) as exc:
                log.append(dict(stage='audit_incomplete',kind=candidate.kind,error_code=type(exc).__name__))
                continue
            log.append(dict(stage='second_confirmation',kind=candidate.kind,**meta,answer=second.model_dump()))
            if second.support!='clear' or max(second.onset_frame,second.evidence_frame)>=len(second_frames):continue
        observations.append(dict(type_id=candidate.kind,timecode_sec=focused[checked.onset_frame]['timestamp_sec'],
                                 evidence_sec=focused[checked.evidence_frame]['timestamp_sec'],visible_fact=checked.visible_fact,
                                 confidence=.8,origin='confirmed_self_hosted_observation',complete_phase_visible=checked.complete_phase_visible))
    if desk=='on' and not observations:
        from service.pipeline.desk import desk_rescan
        observations,details=desk_rescan(frames,deadline,requester=requester)
        log.extend(details)
    if action_windows=='on' and not observations:
        from service.pipeline.action_windows import action_rescan
        observations,details=action_rescan(frames,deadline,requester=requester)
        log.extend(details)
    focus_mode=os.environ.get('AUDITOR_FOCUS','off')
    if focus_mode not in ['off','on']:raise ValueError('Unknown grounded focus mode')
    if focus_mode=='on':
        from service.pipeline.workspace_focus import detail_rescan
        extra,details=detail_rescan(frames,deadline,requester=requester)
        log.extend(details)
        for event in extra:
            if not any(old['type_id']==event['type_id'] and abs(old['timecode_sec']-event['timecode_sec'])<=4 for old in observations):observations.append(event)
    return observations,log


def temporal_windows(frames):
    if not frames:return
    start=frames[0]['timestamp_sec'];last=frames[-1]['timestamp_sec']
    while start<=last:
        window=[f for f in frames if start<=f['timestamp_sec']<=start+12]
        if window:yield window
        if start+12>=last:break
        start+=10


def tiled_observations(frames,deadline,*,requester=None):
    observations=[];log=[];incomplete=False
    for window in temporal_windows(frames):
        if time.monotonic()+25>=deadline:
            incomplete=True;log.append(dict(stage='audit_incomplete',error_code='BudgetLimit'));break
        try:
            found,details=confirmed_observations(window,deadline,requester=requester,overview_count=8,allow_omissions=False)
            observations.extend(found)
            for entry in details:
                log.append(dict(**entry,window_start_sec=window[0]['timestamp_sec'],window_end_sec=window[-1]['timestamp_sec']))
            incomplete|=any(e['stage']=='audit_incomplete' for e in details)
        except (OSError,TimeoutError,ValueError,KeyError,UnusableVisibilityError) as exc:
            incomplete=True;log.append(dict(stage='audit_incomplete',error_code=type(exc).__name__,window_start_sec=window[0]['timestamp_sec'],window_end_sec=window[-1]['timestamp_sec']))
    # Partial positive evidence remains useful; an incomplete negative scan
    # must use the explicitly degraded fallback, never primary clean credit.
    if incomplete and not observations:raise TimeoutError('Incomplete temporal audit without a confirmed observation')
    return observations,log
