"""Source video -> observations -> qualified concerns -> original evidence."""
import base64,hashlib,json,os,time
from pathlib import Path
import cv2
from src import config
from src.video_analysis import _check_deadline


def recipe_fingerprint():
    manifests={'qwen3-8b':'qwen3-vl-8b.json','qwen25-7b':'qwen25-vl-7b.json','qwen3-32b-awq':'qwen3-vl-32b-awq.json'}
    profile=os.environ.get('AUDITOR_PROFILE','qwen3-8b')
    if profile not in manifests:raise ValueError('Unknown pinned auditor profile')
    files=list((config.ROOT/'service/pipeline').glob('*.py'))+[config.ROOT/'service/rules.json',config.ROOT/'deploy'/manifests[profile],
          config.ROOT/'src/video_semantic.py',config.ROOT/'src/video_analysis.py',config.ROOT/'src/video_objects.py',config.ROOT/'scripts/pod_setup.sh']
    value={'sources':{str(p.relative_to(config.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
           'seed':config.SEED,'auditor_profile':profile,'auditor_call_timeout':os.environ.get('AUDITOR_CALL_TIMEOUT','10'),
           'perception_mode':os.environ.get('PERCEPTION_MODE','full'),
           'model_endpoint':os.environ.get('VLLM_URL','http://127.0.0.1:8001')}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def encode_jpeg(image):
    h,w=image.shape[:2]
    if max(h,w)>720:image=cv2.resize(image,(round(w*720/max(h,w)),round(h*720/max(h,w))),interpolation=cv2.INTER_AREA)
    ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,88])
    if not ok:raise ValueError('Evidence encoding failed')
    return jpeg.tobytes()


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
        samples=[dict(timestamp_sec=f['timestamp_sec'],source_frame_index=f['source_frame_index'],image=encode_jpeg(f['image']))
                 for f in iter_frames(path,deadline)]
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
        samples.append(dict(**row,image=encode_jpeg(frame['image'])))
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
    from service.pipeline.auditor import confirmed_observations
    from service.pipeline.fuse import combine
    rules=json.loads((config.ROOT/'service/rules.json').read_text())
    observations,frames=perception(path,deadline-30);perception_sec=time.monotonic()-started
    scene=calibrate(observations);facts=timeline(observations,scene)
    rule_events=evaluate(facts,rules);audits=[];warnings=[]
    try:
        candidates,audits=confirmed_observations(frames,deadline-20,requester=requester)
    except (OSError,TimeoutError,ValueError,KeyError) as exc:
        warnings.append('Auditor unavailable/invalid; weak numeric CPU fallback used: '+type(exc).__name__)
        result=cpu_fallback(path,deadline)
        return result,dict(engine='mission-v1-degraded',recipe_sha256=recipe_fingerprint(),warnings=warnings,sampled_frames=len(frames),perception_sec=perception_sec,elapsed_sec=time.monotonic()-started)
    events=combine(rule_events,candidates,rules);result=build_result(events,frames,rules);_check_deadline(deadline)
    return result,dict(engine='mission-v1',recipe_sha256=recipe_fingerprint(),perception_engine=observations.get('engine'),scene=scene,facts=facts,auditor=audits,events=events,warnings=warnings,
                       sampled_frames=len(frames),perception_sec=perception_sec,elapsed_sec=time.monotonic()-started,production_qualified=False)


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
