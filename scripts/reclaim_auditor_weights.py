"""Remove only hash-verified, inactive, reproducible pinned weight files."""
import argparse,hashlib,json
from pathlib import Path
from scripts.download_auditor import PROFILES
from src import config


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=PROFILES,required=True);p.add_argument('--apply',action='store_true');args=p.parse_args(argv)
    selected=PROFILES[args.profile];root=Path(selected['directory']);manifest=json.loads((config.ROOT/selected['manifest']).read_text())
    if root.parent!=Path('/workspace/models') or root.is_symlink():raise ValueError('Unexpected cache path')
    if json.loads((root/'source-manifest.json').read_text())!=manifest:raise ValueError('Installed cache manifest differs')
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal():continue
        try:command=(process/'cmdline').read_bytes()
        except OSError:continue
        if str(root).encode() in command:raise RuntimeError('Weights still in use')
    assets=[]
    for item in manifest['files']:
        name=item['rfilename']
        if not name.endswith('.safetensors'):continue
        if Path(name).name!=name:raise ValueError('Unexpected weight name')
        path=root/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=item['size']:raise ValueError('Unexpected weight asset')
        assets.append((path,item))
    if not assets:raise ValueError('No pinned weights found')
    if args.apply:
        for path,item in assets:
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                while chunk:=stream.read(8*1024**2):digest.update(chunk)
            if digest.hexdigest()!=item['lfs']['sha256']:raise ValueError('Weight hash differs; nothing removed')
        for path,_ in assets:path.unlink()
    report=dict(profile=args.profile,apply=args.apply,bytes=sum(i['size'] for _,i in assets),files=[p.name for p,_ in assets],raw_files_touched=0,redownload_manifest=selected['manifest'])
    target=config.DATA_PROC/'mission-runtime'/('reclaimed-'+args.profile+'.json');target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
