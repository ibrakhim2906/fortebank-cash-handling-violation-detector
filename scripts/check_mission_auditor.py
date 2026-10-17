"""Dataset-independent color/order and hallucination checks on the Pod."""
import argparse,json,time
from pathlib import Path
from typing import Literal
import numpy as np
from pydantic import BaseModel,ConfigDict
from service.pipeline.analyze import encode_jpeg
from service.pipeline.vlm import ask,frame_limit
from src import config


class Probe(BaseModel):
    model_config=ConfigDict(extra='forbid')
    first_color:Literal['red','blue','gray','other']
    last_color:Literal['red','blue','gray','other']
    person:Literal['yes','no','not_visible']
    cash:Literal['yes','no','not_visible']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--json',type=Path,required=True);p.add_argument('--capacity',action='store_true');args=p.parse_args(argv)
    colors={'red':(0,0,255),'blue':(255,0,0),'gray':(127,127,127)};rows=[]
    for first,last in [('red','blue'),('blue','red'),('gray','gray'),('gray','gray')]:
        frames=[dict(timestamp_sec=i,image=encode_jpeg(np.full((128,128,3),colors[name],np.uint8))) for i,name in enumerate([first,last])]
        start=time.monotonic();row=dict(first=first,last=last,passed=False)
        try:
            answer,meta=ask(frames,'These are synthetic controls. Identify the dominant color of first and last image. Are any actual people or cash/banknotes visible? Do not invent objects from the task context.',Probe,time.monotonic()+60,max_tokens=160)
            row.update(answer=answer.model_dump(),**meta)
            row['passed']=answer.first_color==first and answer.last_color==last and answer.person!='yes' and answer.cash!='yes'
        except Exception as exc:row['error']=type(exc).__name__
        row['elapsed_sec']=time.monotonic()-start;rows.append(row)
    if args.capacity:
        rng=np.random.default_rng(config.SEED);frames=[];count=frame_limit();start=time.monotonic()
        for i in range(count):
            name='red' if i==0 else 'blue' if i==count-1 else 'gray'
            image=np.full((848,464,3),colors[name],np.uint8)
            # Unique synthetic pixels exercise the encoder rather than just
            # repeatedly retrieving one cached image embedding.
            image=np.clip(image.astype(np.int16)+rng.integers(-16,17,image.shape),0,255).astype(np.uint8)
            frames.append(dict(timestamp_sec=i*.5,image=encode_jpeg(image)))
        if len({f['image'] for f in frames})!=count:raise ValueError('Synthetic capacity frames unexpectedly collapsed')
        row=dict(first='red',last='blue',passed=False,capacity_frames=count)
        try:
            answer,meta=ask(frames,'Synthetic capacity test. Identify dominant first/last colors; report actual people or cash only if visible.',Probe,time.monotonic()+60,max_tokens=160)
            row.update(answer=answer.model_dump(),**meta)
            row['passed']=answer.first_color=='red' and answer.last_color=='blue' and answer.person!='yes' and answer.cash!='yes'
        except Exception as exc:row['error']=type(exc).__name__
        row['elapsed_sec']=time.monotonic()-start;rows.append(row)
    value=dict(passed=all(r['passed'] for r in rows),controls=rows,dataset_frames_sent=0)
    args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
    if not value['passed']:raise SystemExit(1)


if __name__=='__main__':main()
