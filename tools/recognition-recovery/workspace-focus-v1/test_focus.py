import base64,importlib.util,json,sys,time
from pathlib import Path
import cv2,numpy as np,pytest
HERE=Path(__file__).parent

def load(name,file):
 s=importlib.util.spec_from_file_location(name,HERE/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

@pytest.fixture
def modules(monkeypatch):
 a=load('focus_auditor','auditor_candidate.py');v=load('focus_vlm','vlm_candidate.py');w=load('focus_detail','workspace_focus_candidate.py')
 monkeypatch.setattr(a,'ask',v.ask);monkeypatch.setattr(w,'ask',v.ask)
 monkeypatch.setitem(sys.modules,'service.pipeline.workspace_focus',w)
 for k,value in dict(AUDITOR_CALL_TIMEOUT='60',AUDITOR_FRAMES='16',AUDITOR_RESOLUTION='standard',AUDITOR_CONFIRMATION='proposal',AUDITOR_SPATIAL='off',AUDITOR_DESK_RESCAN='off',AUDITOR_ACTION_WINDOWS='off',AUDITOR_WORKFLOW='original',AUDITOR_FOCUS='off').items():monkeypatch.setenv(k,value)
 return a,v,w

def frames(n=40):
 image=np.zeros((128,96,3),np.uint8);image[:,30:70]=(10,100,240);ok,jpeg=cv2.imencode('.jpg',image);assert ok
 return [dict(timestamp_sec=i/2,source_frame_index=i*15,image=jpeg.tobytes(),inspection_image=jpeg.tobytes()) for i in range(n)]

def response(answer):return dict(model='qwen-vl',choices=[dict(finish_reason='stop',message=dict(content=json.dumps(answer)))])
def positive(kind='personal_items',met=True):
 return dict(before_fact='Пальто в руках.',visible_fact='Ненадетое пальто положено на рабочий стол.',kind=kind,predicate_met=met,support='clear',counterexamples_excluded=True,before_frame=2,onset_frame=3,evidence_frame=3,context_frame=14)

def requester(items=(),fail_detail=False):
 calls=[]
 def request(url,payload,timeout):
  body=json.loads(payload);calls.append((body,timeout));props=body['response_format']['json_schema']['schema']['properties']
  if 'x0' in props:return response(dict(located=True,consistent_across_views=True,x0=100,y0=100,x1=900,y1=900))
  if 'phases' in props:
   if fail_detail:raise TimeoutError('auxiliary only')
   return response(dict(visibility='clear',worker_and_client_context='Оператор за столом.',phases=[dict(start_frame=0,end_frame=15,visible_action='Работа и смена действия.')],observations=list(items)))
  return response(dict(visibility='clear',observations=[]))
 return request,calls

def test_default_coarse_request_unchanged(modules):
 from service.pipeline import auditor as old
 a,_,_=modules;call,rows=requester()
 old.confirmed_observations(frames(),time.monotonic()+165,requester=call)
 a.confirmed_observations(frames(),time.monotonic()+165,requester=call)
 assert len(rows)==2 and rows[0][0]==rows[1][0]

def test_dynamic_views_preserve_original_clock_and_pixels(modules):
 _,_,w=modules;fs=frames();samples,original=w.focused_views(fs,(100,100,900,900))
 assert len(samples)==16 and len(original)==3
 assert all(b['timestamp_sec']>=a['timestamp_sec'] for a,b in zip(samples,samples[1:]))
 assert all(samples[i]['image']==next(f['inspection_image'] for f in fs if f['timestamp_sec']==samples[i]['timestamp_sec']) for i in original)
 for f in samples:
  image=cv2.imdecode(np.frombuffer(f['image'],np.uint8),cv2.IMREAD_COLOR)
  assert image is not None and len(f['image'])<=1024**2
  if f['focus_view']=='detail':assert image.shape[0]*image.shape[1]<=262144

def test_positive_sources_are_original_api_evidence(modules,monkeypatch):
 from service.pipeline.analyze import build_result
 a,_,w=modules;monkeypatch.setenv('AUDITOR_FOCUS','on');call,rows=requester([positive()]);fs=frames()
 events,logs=a.confirmed_observations(fs,time.monotonic()+165,requester=call)
 assert len(events)==1 and len(rows)==3
 selected,_=w.focused_views(fs,(100,100,900,900));assert events[0]['timecode_sec']==selected[3]['timestamp_sec']
 rules={'types':{'personal_items':{'name':'Вещи','name_en':'Items'}}};result=build_result(events,fs,rules)
 for evidence in result['violations'][0]['evidence_frames']:
  assert base64.b64decode(evidence['image_base64'])==next(f['image'] for f in fs if f['timestamp_sec']==evidence['timestamp_sec'])
 assert 60<rows[-1][1]<=90

def test_normal_clear_observation_never_becomes_alert(modules,monkeypatch):
 a,_,_=modules;monkeypatch.setenv('AUDITOR_FOCUS','on');item=positive(met=False);item['visible_fact']='Личных вещей на столе нет.';call,_=requester([item])
 events,logs=a.confirmed_observations(frames(),time.monotonic()+165,requester=call)
 assert not events

def test_auxiliary_timeout_preserves_successful_baseline(modules,monkeypatch):
 a,_,_=modules;monkeypatch.setenv('AUDITOR_FOCUS','on');call,_=requester(fail_detail=True)
 events,logs=a.confirmed_observations(frames(),time.monotonic()+165,requester=call)
 assert not events and any(c['stage']=='focus_rescan_unresolved' for c in logs)
 assert not any(c['stage']=='audit_incomplete' for c in logs)

@pytest.mark.parametrize('kind',['cash_left_out','counting_without_client'])
def test_crop_cannot_establish_departure_absence(modules,kind):
 _,_,w=modules;item=positive(kind);item['context_frame']=3;call,_=requester([item])
 events,_=w.detail_rescan(frames(),time.monotonic()+165,requester=call)
 assert not events

def test_no_unbudgeted_auxiliary_query(modules):
 _,_,w=modules;events,logs=w.detail_rescan(frames(),time.monotonic()+44,requester=lambda *a:pytest.fail('Too little budget'))
 assert not events and logs[0]['error_code']=='FocusBudget'

def test_replay_clock_includes_detail_budget(modules,monkeypatch):
 a,_,w=modules;monkeypatch.setitem(sys.modules,'service.pipeline.auditor',a)
 b=load('focus_benchmark','benchmark_auditor_candidate.py');real=time.monotonic();original=w.time
 with b.pipeline_replay_clock(lambda:50):assert w.time.monotonic()>=real+49.9
 assert w.time is original
