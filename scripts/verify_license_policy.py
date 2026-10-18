"""Check the delivered runtime against the organisers' licence rule (R50-R56).

Rule: open-source code/models whose licences permit commercial use, each recorded.
Model weights must be Apache-2.0, MIT or BSD. Supporting runtime may also use
weak-copyleft (MPL-2.0, LGPL) and PSF terms, and NVIDIA CUDA runtime terms for
nvidia-* distributions only. Fails closed on GPL/AGPL/SSPL, non-commercial terms,
or any distribution whose licence cannot be identified.
"""
import argparse,collections,json,re,tarfile
from pathlib import Path
from src import config

INVENTORY=config.DATA_PROC/'mission-v6-bootstrap/pod-installed-licenses.json'
INTEGRITY=config.DATA_PROC/'recognition-recovery-20261018/licenses/verification.json'
ARCHIVE=config.DATA_PROC/'mission-license-notices-20261018.tar.gz'
MODEL=config.ROOT/'deploy/qwen3-vl-32b-awq.json'
DECODER=config.ROOT/'tools/local-video-decoder/manifest.json'
OUTPUT=config.DATA_PROC/'license-policy-verification.json'
# Declarations missing or 'UNKNOWN' in metadata, resolved from the preserved notice text.
NOTICE_RESOLVED={'xformers':('BSD-3-Clause','BSD 3-Clause License'),'openai-harmony':('Apache-2.0','Apache License'),
                 'setuptools':('MIT','Permission is hereby granted, free of charge')}
MODEL_ALLOWED={'apache-2.0','mit','bsd-2-clause','bsd-3-clause'}
DISALLOWED=re.compile(r'AGPL|Affero|SSPL|(?<![AL])GPL|GNU General Public|non-?commercial|CC-BY-NC',re.I)
FAMILIES=[('weak_copyleft',re.compile(r'LGPL|Lesser General Public|MPL|Mozilla',re.I)),
          ('nvidia_cuda',re.compile(r'NVIDIA|Proprietary',re.I)),
          ('psf',re.compile(r'PSF|Python Software Foundation|CNRI',re.I)),
          ('permissive',re.compile(r'MIT|BSD|Apache|ISC|Unlicense|CC0|Zlib',re.I))]


def declaration(row):
    parts=[row.get('expression') or '',*(row.get('license_classifiers') or []),str(row.get('license_metadata') or '')[:120]]
    return ' ; '.join(p for p in parts if p and p!='UNKNOWN')


def classify(row,notices):
    """Return (family, declaration) or (None, reason) for one installed distribution."""
    text=declaration(row)
    if row['name'] in NOTICE_RESOLVED:
        spdx,marker=NOTICE_RESOLVED[row['name']]
        if not any(marker in notices.get(f,'') for f in notices if row['name'].replace('-','_') in f):return None,'notice text not found'
        text=(text+' ; ' if text else '')+spdx+' (notice)'
    if not text:return None,'no licence declaration'
    if DISALLOWED.search(text.replace('LGPL','').replace('Lesser General Public','')):return None,'disallowed: '+text
    found=[name for name,pattern in FAMILIES if pattern.search(text)]
    if not found:return None,'unrecognised: '+text
    if 'nvidia_cuda' in found and not row['name'].startswith('nvidia-'):return None,'proprietary terms outside NVIDIA CUDA runtime: '+text
    return found[0],text


def notice_texts(names):
    texts={}
    with tarfile.open(ARCHIVE) as archive:
        for member in archive:
            if member.isfile() and any(n.replace('-','_') in member.name for n in names):
                texts[member.name]=archive.extractfile(member).read(65536).decode(errors='replace')
    return texts


def verify():
    integrity=json.loads(INTEGRITY.read_text());inventory=json.loads(INVENTORY.read_text())
    model=json.loads(MODEL.read_text());decoder=json.loads(DECODER.read_text())
    notices=notice_texts(NOTICE_RESOLVED)
    rows=[];failures=[]
    for row in inventory['distributions']:
        family,detail=classify(row,notices)
        rows.append(dict(environment=row['environment'],name=row['name'],version=row['version'],family=family,declaration=detail))
        if family is None:failures.append(f"{row['environment']}/{row['name']} {row['version']}: {detail}")
    extra=[dict(environment='runtime',name='CPython',version=inventory['python'].split()[0],family='psf',declaration='PSF-2.0'),
           dict(environment='decoder',name='FFmpeg/ffprobe',version=decoder['release'],family='weak_copyleft',declaration=decoder['license'])]
    if not re.fullmatch(r'LGPL-[0-9.]+(-or-later|-only)?',decoder['license']):failures.append('decoder licence is not LGPL: '+decoder['license'])
    model_ok=model.get('license','').lower() in MODEL_ALLOWED and integrity.get('model_card_declaration','').lower()==model.get('license','').lower()
    if not model_ok:failures.append('model weights not Apache-2.0/MIT/BSD on manifest and pinned card')
    prerequisites=dict(evidence_integrity_passed=integrity.get('evidence_integrity_passed') is True and not integrity.get('failures'),
                       package_notice_presence_complete=integrity.get('package_notice_presence_complete') is True,
                       distribution_count_matches=len(inventory['distributions'])==integrity.get('distribution_count'))
    failures+=[k for k,v in prerequisites.items() if not v]
    counts=collections.Counter(r['family'] for r in rows+extra)
    return dict(policy='organisers R50-R56: open source, commercial use permitted; model Apache/MIT/BSD; runtime may add MPL/LGPL/PSF and NVIDIA CUDA terms',
                scope='Delivered service: API service env, model-server env, CPython, FFmpeg decoder, model weights. Host driver/base image are Runpod platform components outside this table.',
                model=dict(repo=model['repo'],revision=model['revision'],license=model['license'],passed=model_ok),
                prerequisites=prerequisites,family_counts=dict(counts),distributions=len(rows),failures=failures,
                license_table_verified=not failures,rows=rows+extra)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUTPUT);args=p.parse_args(argv)
    value=verify();args.out.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:value[k] for k in ['license_table_verified','distributions','family_counts','failures','model']},ensure_ascii=False))
    raise SystemExit(0 if value['license_table_verified'] else 1)


if __name__=='__main__':main()
