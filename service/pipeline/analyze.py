"""Source video -> observations -> qualified concerns -> original evidence."""
import base64,hashlib,json,os,time
from pathlib import Path
import cv2
from src import config
from src.video_analysis import _check_deadline


def recipe_fingerprint():
    manifests={'qwen3-8b':'qwen3-vl-8b.json','qwen25-7b':'qwen25-vl-7b.json','qwen3-32b-awq':'qwen3-vl-32b-awq.json',
               'qwen3-30b-a3b-awq':'qwen3-vl-30b-a3b-awq.json'}
    profile=os.environ.get('AUDITOR_PROFILE','qwen3-8b')
    if profile not in manifests:raise ValueError('Unknown pinned auditor profile')
    files=list((config.ROOT/'service/pipeline').glob('*.py'))+[config.ROOT/'service/rules.json',config.ROOT/'deploy'/manifests[profile],
          config.ROOT/'src/video_semantic.py',config.ROOT/'src/video_analysis.py',config.ROOT/'src/video_objects.py',config.ROOT/'scripts/pod_setup.sh']
    value={'sources':{str(p.relative_to(config.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
           'seed':config.SEED,'auditor_profile':profile,'auditor_call_timeout':os.environ.get('AUDITOR_CALL_TIMEOUT','10'),
           'perception_mode':os.environ.get('PERCEPTION_MODE','full'),
           'auditor_resolution':os.environ.get('AUDITOR_RESOLUTION','standard'),
           'auditor_scan':os.environ.get('AUDITOR_SCAN','full'),
           'auditor_spatial':os.environ.get('AUDITOR_SPATIAL','off'),
           'auditor_desk_rescan':os.environ.get('AUDITOR_DESK_RESCAN','off'),
           'auditor_action_windows':os.environ.get('AUDITOR_ACTION_WINDOWS','off'),
           'auditor_workflow':os.environ.get('AUDITOR_WORKFLOW','original'),
           'auditor_confirmation':os.environ.get('AUDITOR_CONFIRMATION','strict'),
           'auditor_frames':os.environ.get('AUDITOR_FRAMES','16'),
           'auditor_context':os.environ.get('AUDITOR_CONTEXT','16384'),
           'model_endpoint':os.environ.get('VLLM_URL','http://127.0.0.1:8001')}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def encode_jpeg(image):
    mode=os.environ.get('AUDITOR_RESOLUTION','standard')
    if mode not in ['standard','source']:raise ValueError('Unknown auditor image resolution')
    edge=1024 if mode=='source' else 720
    h,w=image.shape[:2]
    if max(h,w)>edge:image=cv2.resize(image,(round(w*edge/max(h,w)),round(h*edge/max(h,w))),interpolation=cv2.INTER_AREA)
    ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,88])
    if not ok:raise ValueError('Evidence encoding failed')
    return jpeg.tobytes()


def encode_inspection_jpeg(image):
    """Bounded source-detail JPEG for opt-in crops, never public/coarse evidence."""
    h,w=image.shape[:2]
    for edge in [2048,1536,1024]:
        ratio=min(1.,edge/max(h,w));sample=image if ratio==1. else cv2.resize(image,(max(1,round(w*ratio)),max(1,round(h*ratio))),interpolation=cv2.INTER_AREA)
        for quality in [88,75,60]:
            ok,jpeg=cv2.imencode('.jpg',sample,[cv2.IMWRITE_JPEG_QUALITY,quality])
            if not ok:raise ValueError('Inspection encoding failed')
            if jpeg.nbytes<=1024**2:return jpeg.tobytes()
    raise ValueError('Inspection JPEG exceeds bounded frame budget')


def source_sample(frame,index):
    sample=dict(timestamp_sec=frame['timestamp_sec'],source_frame_index=frame['source_frame_index'],image=encode_jpeg(frame['image']))
    if os.environ.get('AUDITOR_SPATIAL','off') in ['open_state','state_ledger','storage_veto'] or os.environ.get('AUDITOR_DESK_RESCAN','off')=='on':
        # The service accepts at most300s, sampled at2fps. Keep bytes only;
        #602samples allow timestamp rounding and cap auxiliary retention.
        if index>=602:raise ValueError('Spatial inspection sample budget exceeded')
        sample['inspection_image']=encode_inspection_jpeg(frame['image'])
    return sample


def cpu_fallback(path,deadline):
    from src.video_analysis import analyze_video
    artifact=config.DATA_PROC/'video-baseline-comparison/models/baseline.joblib'
    return analyze_video(path,model_path=artifact,deadline=deadline)


def perception(path,deadline):
    from service.pipeline.frames import iter_frames
    mode=os.environ.get('PERCEPTION_MODE','full')
    if mode not in ['full','frames_only']:raise ValueError('Unknown perception mode')
    if mode=='frames_only':
        rules=json.loads((config.ROOT/'service/rules.json').read_text())
        if any(spec.get('enabled') for spec in rules['types'].values()):
            raise ValueError('Frame-only mode cannot bypass an enabled object rule')
        samples=[source_sample(frame,index) for index,frame in enumerate(iter_frames(path,deadline))]
        # No detector was run. Empty *observations* means no factual claims;
        # never manufacture a zero-occupancy or absence timeline from it.
        return dict(frames=[],engine='source_frames_only_no_object_observations'),samples
    from service.pipeline.detector import get_detector,_VERSION
    from service.pipeline.tracker import Tracker,foreground_people
    detector=get_detector();tracker=Tracker();rows=[];samples=[]
    for frame in iter_frames(path,deadline):
        detections=tracker.update(detector.detect(frame['image'],deadline),frame['timestamp_sec'])
        row={k:frame[k] for k in ['timestamp_sec','source_frame_index']}
        rows.append(dict(**row,detections=detections,foreground_person_tracks=[d['track_id'] for d in foreground_people(detections)]))
        samples.append(source_sample(frame,len(samples)))
    return dict(frames=rows,engine=_VERSION),samples


def build_result(events,frames,rules):
    result=[]
    for event in events:
        spec=rules['types'][event['type_id']]
        first=min(frames,key=lambda f:abs(f['timestamp_sec']-event['timecode_sec']))
        support=min(frames,key=lambda f:abs(f['timestamp_sec']-event.get('evidence_sec',event['timecode_sec']+1.5)))
        evidence=[first]+([support] if support['source_frame_index']!=first['source_frame_index'] else [])
        fact=event.get('visible_fact','Наблюдается сочетание физических признаков; требуется проверка контекста.')
        context=spec.get('context_source')
        source=(f"Нормативный контекст: {context['act']} ({context['url']}); {context['scope']}" if context else
                'Прямая публичная норма для этого наблюдения не подтверждена; внутренний регламент банка не установлен.')
        description=f"На {first['timestamp_sec']:.2f} с видно: {fact} {source} Применимость к операции требует проверки человеком."
        result.append(dict(type=spec['name']+' / '+spec['name_en'],timecode_sec=first['timestamp_sec'],description=description,
                           evidence_frames=[dict(timestamp_sec=f['timestamp_sec'],image_base64=base64.b64encode(f['image']).decode()) for f in evidence]))
    return dict(verdict='violation' if result else 'no_violation',violations=result)


def analyze_detailed(path,deadline=None,*,requester=None):
    started=time.monotonic();deadline=min(deadline or started+185,started+185)
    from service.pipeline.scene import calibrate
    from service.pipeline.facts import timeline
    from service.pipeline.engine import evaluate
    from service.pipeline.auditor import confirmed_observations,tiled_observations,UnusableVisibilityError
    from service.pipeline.fuse import combine
    rules=json.loads((config.ROOT/'service/rules.json').read_text())
    observations,frames=perception(path,deadline-30);perception_sec=time.monotonic()-started
    scene=calibrate(observations);facts=timeline(observations,scene)
    rule_events=evaluate(facts,rules);audits=[];warnings=[]
    try:
        mode=os.environ.get('AUDITOR_SCAN','full')
        if mode not in ['full','tiled']:raise ValueError('Unknown temporal scan mode')
        candidates,audits=(tiled_observations if mode=='tiled' else confirmed_observations)(frames,deadline-20,requester=requester)
        if not candidates and not rule_events and any(e['stage']=='audit_incomplete' for e in audits):
            raise TimeoutError('Incomplete audit without confirmed positive evidence')
    except (OSError,TimeoutError,ValueError,KeyError,UnusableVisibilityError) as exc:
        warnings.append('Auditor unavailable/invalid; weak numeric CPU fallback used: '+type(exc).__name__)
        result=cpu_fallback(path,deadline)
        failure=next((entry for entry in reversed(audits) if entry.get('stage')=='audit_incomplete'),{})
        return result,dict(engine='mission-v1-degraded',recipe_sha256=recipe_fingerprint(),warnings=warnings,
                           auditor=audits,failed_stage=failure.get('failed_stage','auditor_dispatch'),
                           error_code=failure.get('error_code',type(exc).__name__),fallback_exception_type=type(exc).__name__,
                           sampled_frames=len(frames),perception_sec=perception_sec,elapsed_sec=time.monotonic()-started)
    complete=not any(e['stage']=='audit_incomplete' for e in audits)
    if not complete:warnings.append('Only supported observations are returned; the temporal audit was incomplete.')
    events=combine(rule_events,candidates,rules);result=build_result(events,frames,rules);_check_deadline(deadline)
    return result,dict(engine='mission-v1',recipe_sha256=recipe_fingerprint(),perception_engine=observations.get('engine'),scene=scene,facts=facts,auditor=audits,events=events,warnings=warnings,
                       sampled_frames=len(frames),perception_sec=perception_sec,elapsed_sec=time.monotonic()-started,audit_complete=complete,production_qualified=False)


def analyze(path,deadline=None):
    deadline=deadline or time.monotonic()+200
    if os.environ.get('PIPELINE_STUB')=='1':
        from src.video_semantic import sample_video
        sample_video(path,deadline);return dict(verdict='no_violation',violations=[])
    if os.environ.get('AUDITOR','off')!='on':return cpu_fallback(path,deadline)
    result,metadata=analyze_detailed(path,deadline)
    # Only the job worker opts into metadata writing. Never write alongside
    # supplied videos, derived source inventory, or arbitrary caller paths.
    jobs=os.environ.get('JOBS_DIR');parent=Path(path).resolve().parent
    if jobs and parent.parent==Path(jobs).resolve():
        from service.app import atomic_json
        atomic_json(parent/'analysis.json',metadata)
    return result
