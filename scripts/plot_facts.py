"""Render private development-only observation strips with truth anchors."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.eval_video import PROTOCOL,ANNOTATIONS,SPLIT
from service.pipeline.scene import calibrate
from service.pipeline.facts import timeline
from src import config


def main():
    protocol=json.loads(PROTOCOL.read_text());split=json.loads(SPLIT.read_text());notes=json.loads(ANNOTATIONS.read_text())
    groups=[g for g in protocol['groups'] if g['group_id'] in split['dev']]
    lookalikes={'Х-3','Х-25','Х-27','Х-29','Х-35'}
    positive=[g for g in groups if g['target']][:6]
    clean=sorted([g for g in groups if not g['target']],key=lambda g:(not bool(set(g['video_ids'])&lookalikes),g['group_id']))[:6]
    out=config.ROOT/'reports/facts';out.mkdir(parents=True,exist_ok=True);summary=[]
    keys=['staff_candidate_observed','staff_present','cash_on_desk','safe_open','door_open','outsider_present','phone_near_staff','book_near_staff']
    for g in positive+clean:
        cached=json.loads((config.DATA_PROC/'mission-detections-v1'/(g['group_id']+'.json')).read_text())
        rows=timeline(cached,calibrate(cached));values=np.array([[1 if r['facts'].get(k) is True else 0 if r['facts'].get(k) is False else .5 for r in rows] for k in keys])
        fig,ax=plt.subplots(figsize=(12,3.5));ax.imshow(values,aspect='auto',vmin=0,vmax=1,cmap='PiYG',extent=[0,rows[-1]['timestamp_sec'],len(keys)-.5,-.5],interpolation='nearest')
        ax.set_yticks(range(len(keys)),keys);ax.set_xlabel('Source seconds; gray = unknown, green = positive observation')
        for e in notes['events']:
            if e['group_id']==g['group_id']:ax.axvline(e['timecode_sec'],color='red',lw=1)
        ax.set_title(','.join(g['video_ids'])+' — development observation diagnostic');fig.tight_layout();fig.savefig(out/(g['group_id']+'.png'));plt.close(fig)
        summary.append(dict(group_id=g['group_id'],aliases=g['video_ids'],compliance_states_known=sum(r['facts'].get(k) is not None for r in rows for k in ['cash_on_desk','safe_open','door_open','outsider_present'])))
    target=config.DATA_PROC/'mission-v6-bootstrap/fact-strip-summary.json';target.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(strips=len(summary),all_development_only=True,compliance_predicates_not_established=all(r['compliance_states_known']==0 for r in summary))))


if __name__=='__main__':main()
