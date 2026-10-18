import difflib,hashlib,json
from pathlib import Path
ROOT=Path('/home/ibrak/datathon-kit');HERE=Path(__file__).resolve().parent;changes={}
rel='service/pipeline/auditor.py';before=(ROOT/rel).read_text();needle='    return observations,log\n\n\ndef temporal_windows'
assert before.count(needle)==1
replacement="""    focus_mode=os.environ.get('AUDITOR_FOCUS','off')
    if focus_mode not in ['off','on']:raise ValueError('Unknown grounded focus mode')
    if focus_mode=='on':
        from service.pipeline.workspace_focus import detail_rescan
        extra,details=detail_rescan(frames,deadline,requester=requester)
        log.extend(details)
        for event in extra:
            if not any(old['type_id']==event['type_id'] and abs(old['timecode_sec']-event['timecode_sec'])<=4 for old in observations):observations.append(event)
"""+needle
changes[rel]=(before,before.replace(needle,replacement))
rel='service/pipeline/analyze.py';before=(ROOT/rel).read_text();needle="'auditor_spatial':os.environ.get('AUDITOR_SPATIAL','off'),";assert needle in before
after=before.replace(needle,needle+"\n           'auditor_focus':os.environ.get('AUDITOR_FOCUS','off'),")
after=after.replace("or os.environ.get('AUDITOR_DESK_RESCAN','off')=='on':","or os.environ.get('AUDITOR_DESK_RESCAN','off')=='on' or os.environ.get('AUDITOR_FOCUS','off')=='on':")
changes[rel]=(before,after)
rel='service/pipeline/vlm.py';before=(ROOT/rel).read_text();after=before.replace('requester=None,max_tokens=768):','requester=None,max_tokens=768,call_timeout=None):').replace("float(os.environ.get('AUDITOR_CALL_TIMEOUT','10'))","float(os.environ.get('AUDITOR_CALL_TIMEOUT','10') if call_timeout is None else call_timeout)");assert after!=before;changes[rel]=(before,after)
rel='scripts/benchmark_auditor.py';before=(ROOT/rel).read_text();needle="'service.pipeline.action_windows']:";assert needle in before;changes[rel]=(before,before.replace(needle,"'service.pipeline.action_windows','service.pipeline.workspace_focus']:"))
changes['service/pipeline/workspace_focus.py']=('',(HERE/'workspace_focus.py').read_text())
parts=[]
for rel,(before,after) in changes.items():
 compile(after,rel,'exec');(HERE/(Path(rel).stem+'_candidate.py')).write_text(after)
 parts.extend(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),fromfile='a/'+rel if before else '/dev/null',tofile='b/'+rel))
patch=''.join(parts);(HERE/'workspace-focus.patch').write_text(patch)
print(json.dumps({'private_candidate_files':len(changes),'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'live_source_changed':False,'model_calls':0}))
