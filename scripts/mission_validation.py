"""Evaluation-only data adapters and deterministic robustness transforms."""
import hashlib,json,re,subprocess
from pathlib import Path
import cv2
import pandas as pd
from src import config

TRANSFORMS={'flip':'hflip','crop':'crop=trunc(iw*0.88/2)*2:trunc(ih*0.88/2)*2,scale=trunc(iw/0.88/2)*2:trunc(ih/0.88/2)*2',
            'rotate_plus':'rotate=10*PI/180:fillcolor=black','rotate_minus':'rotate=-10*PI/180:fillcolor=black',
            'dark':'colorchannelmixer=rr=0.65:gg=0.65:bb=0.65','resolution':'scale=trunc(iw*0.75/2)*2:trunc(ih*0.75/2)*2','reencode':'null'}


def alias_number(value):
    value=Path(str(value)).stem
    match=re.fullmatch(r'(?:X|Х|#U0425)[- _]?(\d+)',value,re.IGNORECASE)
    if not match:raise ValueError('Unrecognized source alias')
    return int(match[1])


def seconds(value):
    if pd.isna(value) or not str(value).strip():return None
    if isinstance(value,(int,float)):return float(value)
    text=str(value).strip().lower(); match=re.fullmatch(r'(?:(\d+)\s*мин\w*\s*)?(?:(\d+)\s*сек\w*\s*)?',text)
    if not match or not any(match.groups()):raise ValueError('Unrecognized label timestamp')
    return 60*int(match[1] or 0)+int(match[2] or 0)


def external_groups(videos,labels):
    """Read only organizer first three columns, deduplicate by full bytes."""
    frame=pd.read_excel(labels) if labels.suffix.lower() in ['.xlsx','.xls'] else pd.read_csv(labels)
    if len(frame.columns)<3:raise ValueError('Expected organizer three-column labels')
    paths={}
    for path in videos.rglob('*'):
        if path.suffix.lower() not in ['.mov','.mp4']:continue
        number=alias_number(path.name)
        if number in paths:raise ValueError('Ambiguous alias number')
        paths[number]=path
    groups={}
    for alias,stamp,label in frame.iloc[:,:3].itertuples(index=False,name=None):
        path=paths[alias_number(alias)];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        text=str(label).strip().lower()
        if text not in ['нарушение','без нарушения','0','1']:raise ValueError('Unknown binary label')
        target=int(text in ['нарушение','1']);timestamp=seconds(stamp)
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps if fps>0 else 0;cap.release()
        if duration<=0 or (target and (timestamp is None or not 0<=timestamp<=duration)):raise ValueError('Invalid source timing')
        g=groups.setdefault(digest,dict(group_id=digest,sha256=digest,video_ids=[],paths=[],path=str(path.resolve()),target=target,duration_sec=duration,timestamp_annotations=[]))
        if g['target']!=target:raise ValueError('Conflicting duplicate labels')
        g['video_ids'].append(Path(str(alias)).stem);g['paths'].append(str(path.resolve()))
        g['timestamp_annotations'].append(dict(video_id=str(alias),timestamp_sec=timestamp))
    return list(groups.values())


def augment(path,name,destination):
    if name not in TRANSFORMS:raise ValueError('Unknown augmentation')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    identity={'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'filter':TRANSFORMS[name],'codec':'libopenh264','bitrate':'1800k','pixel_format':'yuv420p'}
    metadata=destination.with_suffix('.provenance.json')
    if destination.exists():
        if not metadata.exists() or json.loads(metadata.read_text())!=identity:raise ValueError('Augmentation provenance differs')
        return destination
    ffmpeg=config.ROOT/'tools/local-video-decoder/bin/ffmpeg'
    temporary=destination.with_suffix('.partial.mp4')
    subprocess.run([str(ffmpeg),'-nostdin','-v','error','-i',str(path),'-an','-vf',TRANSFORMS[name],'-c:v','libopenh264','-b:v','1800k','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(temporary)],check=True,timeout=180)
    temporary.chmod(0o600);temporary.replace(destination);metadata.write_text(json.dumps(identity,sort_keys=True)+'\n');metadata.chmod(0o600);return destination


def event_signature(result):
    return result.get('verdict'),sorted((v['type'],round(v['timecode_sec'],1)) for v in result.get('violations',[]))
