"""Reclaim only pinned, inactive, reproducible old GGUF assets on this Pod."""
import argparse,hashlib,json
from pathlib import Path
from src import config


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');args=p.parse_args(argv)
    plan=json.loads((config.ROOT/'deploy/runpod-assets.json').read_text());assets=[]
    for name in ['30b','8b']:
        for asset in plan['models'][name]['assets']:
            path=config.ROOT/asset['path']
            if path.suffix!='.gguf':continue
            expected=config.ROOT/'models/local-action-model'/('runpod-'+name+'-v1')
            if any(parent.is_symlink() for parent in [path,*path.parents] if parent.is_relative_to(config.ROOT)) or not path.resolve().is_relative_to(expected.resolve()):raise ValueError('Unexpected weight path')
            if path.exists():assets.append((path,asset))
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal():continue
        try:command=(process/'cmdline').read_bytes()
        except OSError:continue
        if any(str(path).encode() in command for path,_ in assets):raise RuntimeError('Old model still in use; no files removed')
    report=dict(apply=args.apply,files=[str(p.relative_to(config.ROOT)) for p,_ in assets],bytes=sum(p.stat().st_size for p,_ in assets),raw_files_touched=0)
    if args.apply:
        # Validate every file before removing any; retain manifests/licenses and
        # exact public redownload URLs. Never traverse a model directory tree.
        for path,asset in assets:
            if path.stat().st_size!=asset['size']:raise ValueError('Old asset size changed')
            digest=hashlib.sha256()
            with path.open('rb') as f:
                while block:=f.read(8*1024**2):digest.update(block)
            if digest.hexdigest()!=asset['sha256']:raise ValueError('Old asset hash changed')
        for path,_ in assets:path.unlink()
        target=config.DATA_PROC/'mission-runtime/old-weight-reclamation.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
