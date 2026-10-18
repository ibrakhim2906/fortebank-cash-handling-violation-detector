"""Download only pinned, hash-checked Qwen Apache2.0 files; no remote code."""
import argparse,hashlib,json,os,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO='Qwen/Qwen3-VL-8B-Instruct'
REVISION='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
PROFILES={
    'qwen3-8b':dict(repo=REPO,revision=REVISION,manifest='deploy/qwen3-vl-8b.json',directory='/workspace/models/qwen3-vl-8b',architecture='Qwen3VLForConditionalGeneration'),
    'qwen25-7b':dict(repo='Qwen/Qwen2.5-VL-7B-Instruct',revision='cc594898137f460bfe9f0759e9844b3ce807cfb5',manifest='deploy/qwen25-vl-7b.json',directory='/workspace/models/qwen25-vl-7b',architecture='Qwen2_5_VLForConditionalGeneration'),
    'qwen3-32b-awq':dict(repo='QuantTrio/Qwen3-VL-32B-Instruct-AWQ',revision='c2d44b1376c73fa120fdb4e216c2ef61c1945498',manifest='deploy/qwen3-vl-32b-awq.json',directory='/workspace/models/qwen3-vl-32b-awq',architecture='Qwen3VLForConditionalGeneration'),
    'qwen3-30b-a3b-awq':dict(repo='QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ',revision='a5ea10732e2c9330957864ece3ec66d806a4b00d',manifest='deploy/qwen3-vl-30b-a3b-awq.json',directory='/workspace/models/qwen3-vl-30b-a3b-awq',architecture='Qwen3VLMoeForConditionalGeneration'),
}


def install(manifest,root):
    profile=next((p for p in PROFILES.values() if manifest['repo']==p['repo'] and manifest['revision']==p['revision']),None)
    if profile is None or manifest['license'] not in ['apache-2.0','mit']:raise ValueError('Wrong auditor manifest')
    if root.is_symlink():raise ValueError('Model directory cannot be a symlink')
    root.mkdir(parents=True,exist_ok=True)
    def one(item):
        name=item['rfilename']
        if name in ['','.','..'] or Path(name).name!=name or '\\' in name:raise ValueError('Unsafe model filename')
        target=root/name;temporary=root/(name+'.partial');expected=item.get('lfs',{}).get('sha256')
        if target.is_symlink() or temporary.is_symlink():raise ValueError('Model assets cannot be symlinks')
        def digest(path):
            h=hashlib.sha256()
            with path.open('rb') as f:
                while b:=f.read(8*1024**2):h.update(b)
            return h.hexdigest()
        def valid(path):
            if path.stat().st_size!=item['size']:return False
            if expected:return digest(path)==expected
            h=hashlib.sha1();h.update(('blob '+str(item['size'])+'\0').encode())
            with path.open('rb') as f:
                while b:=f.read(8*1024**2):h.update(b)
            return h.hexdigest()==item['blobId']
        if target.exists():
            if not valid(target):raise ValueError('Existing asset differs')
            return
        url=f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{name}"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=180) as response,temporary.open('wb') as out:
                    while b:=response.read(8*1024**2):out.write(b)
                if not valid(temporary):raise ValueError('Download hash/size mismatch')
                temporary.replace(target);print(json.dumps(dict(file=name,bytes=item['size'],sha256=digest(target))),flush=True);return
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt==2:raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(one,manifest['files']))
    configuration=json.loads((root/'config.json').read_text())
    if configuration.get('architectures')!=[profile['architecture']] or configuration.get('auto_map'):
        raise ValueError('Unexpected/custom model architecture')
    (root/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(model=manifest['repo'],revision=manifest['revision'],installed=True)))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=PROFILES,default='qwen3-8b');args=p.parse_args(argv)
    selected=PROFILES[args.profile];manifest=json.loads(Path(selected['manifest']).read_text());root=Path(selected['directory'])
    install(manifest,root)


if __name__=='__main__':main()
