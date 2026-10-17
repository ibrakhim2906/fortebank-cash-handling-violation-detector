import json
from scripts.eval_video import freeze_split,match_events,score_records
from scripts.rubric_score import calculate


def test_matching_is_one_to_one_and_maximizes_hits():
    truth=[{'timecode_sec':0},{'timecode_sec':2}]
    pred=[{'timecode_sec':1},{'timecode_sec':0}]
    assert len(match_events(truth,pred,1))==2
    assert len(match_events(truth,pred[:1],2))==1


def test_failures_are_not_clean_and_events_cannot_be_spammed():
    groups=[dict(group_id='a',target=1,video_ids=['X-1']),dict(group_id='b',target=0,video_ids=['X-2'])]
    events=[dict(group_id='a',timecode_sec=10,type_ids=['v'],flags=[])]
    result=dict(verdict='violation',violations=[dict(type='v',timecode_sec=10)]*3)
    records=[dict(group_id='a',status='done',result=result,elapsed_sec=2),dict(group_id='b',status='failed',elapsed_sec=3)]
    scored=score_records(records,groups,events,{'v':dict(name='V',name_en='V')})
    assert scored['accuracy']==.5 and scored['failed']==1 and scored['tn']==0
    assert scored['extra_events']==2 and scored['event_precision']==1/3


def test_split_is_frozen_group_disjoint_and_actual_setup_only(tmp_path):
    groups=[dict(group_id=str(i),target=i%2,video_ids=['X-'+str(i+1)]) for i in range(40)]
    path=tmp_path/'split.json';a=freeze_split(groups,path);assert a==freeze_split(groups,path)
    assert len(a['reserve'])==10 and set(a['reserve']).isdisjoint(a['dev'])
    seen=[]
    for fold in a['folds']:
        assert set(fold['train']).isdisjoint(fold['validation']);seen+=fold['validation']
    assert sorted(seen)==sorted(a['dev'])


def test_proxy_requires_measured_gates_and_honest_provenance():
    excellent=dict(accuracy=1,hit_1s=1,hit_2s=1,type_accuracy=1,recall=1,fp=0)
    assert calculate(excellent)['estimated_proxy_score']==0
    excellent.update(validation_kind='leave-one-setup-out',fold_integrity_verified=True,mode='full')
    value=calculate(excellent)
    assert value['estimated_proxy_score']==55 and value['components']['product']==0
    api=dict(mode='stub',n_videos=40,completed=40,failed=0,worst_sec=1)
    assert calculate(excellent,api=api,gates={'pytest_green':True})['components']['product']==0


def test_crossfit_offsets_never_use_validation_labels():
    from scripts.mission_crossfit import crossfit
    groups=[dict(group_id=str(i),target=1,video_ids=['X-'+str(i+1)],duration_sec=100) for i in range(6)]
    events=[dict(group_id=str(i),timecode_sec=11 if i<3 else 29,type_ids=['v'],flags=[]) for i in range(6)]
    records=[dict(group_id=str(i),status='done',result={'verdict':'violation','violations':[{'type':'v','timecode_sec':10 if i<3 else 30}]},elapsed_sec=1) for i in range(6)]
    split=dict(dev=[str(i) for i in range(6)],folds=[dict(name='a',train=['0','1','2'],validation=['3','4','5']),dict(name='b',train=['3','4','5'],validation=['0','1','2'])])
    notes=dict(events=events,taxonomy={'v':dict(name='V',name_en='V')})
    value=crossfit({'records':records,'mode':'full','oof_eligibility_verified':True},{'groups':groups},notes,split,True)
    assert value['folds'][0]['offsets']=={'v':1} and value['folds'][1]['offsets']=={'v':-1}
    assert [r['result']['violations'][0]['timecode_sec'] for r in value['records']]==[31,31,31,9,9,9]


def test_real_api_schema_positive_is_scored():
    import base64,cv2,numpy as np
    from src.video_api import Result
    ok,jpeg=cv2.imencode('.jpg',np.zeros((16,16,3),dtype=np.uint8));assert ok
    result=Result.model_validate(dict(verdict='violation',violations=[dict(type='v',timecode_sec=10,
        description='Observable evidence for human review.',evidence_frames=[dict(timestamp_sec=10,image_base64=base64.b64encode(jpeg).decode())])])).model_dump()
    group=dict(group_id='a',target=1,video_ids=['X-1'])
    scored=score_records([dict(group_id='a',status='done',result=result,elapsed_sec=1)],[group],
        [dict(group_id='a',timecode_sec=10,type_ids=['v'],flags=[])],{'v':dict(name='V',name_en='V')})
    assert scored['accuracy']==scored['hit_1s']==scored['type_accuracy']==1


def test_nonfinite_or_percentage_metrics_do_not_earn_full_proxy_points():
    invalid=dict(accuracy=float('nan'),hit_1s=float('inf'),hit_2s=100,type_accuracy=True,
                 validation_kind='leave-one-setup-out',fold_integrity_verified=True,mode='api')
    value=calculate(invalid,api=dict(mode='api',n_videos=1,completed=1,failed=0,worst_sec=-1),gates={'pytest_green':True})
    assert value['estimated_proxy_score']==0


def test_single_smoke_job_cannot_earn_full_product_points():
    report=dict(mode='api',subset='dev',n_videos=1,completed=1,failed=0,worst_sec=1)
    assert calculate({},api=report,gates={'pytest_green':True})['components']['product']==0
    report.update(subset='all',n_videos=40,completed=40)
    assert calculate({},api=report,gates={'pytest_green':True})['components']['product']==10
