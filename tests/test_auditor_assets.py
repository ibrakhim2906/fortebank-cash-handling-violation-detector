import json
from pathlib import Path
import pytest
from scripts.download_auditor import install,PROFILES


def test_only_pinned_permissive_profiles_are_allowed(tmp_path):
    with pytest.raises(ValueError):install(dict(repo='unknown/remote-code',revision='unknown',license='apache-2.0'),tmp_path)
    p=PROFILES['qwen3-8b']
    with pytest.raises(ValueError):install(dict(repo=p['repo'],revision=p['revision'],license='research-only'),tmp_path)


def test_manifest_pins_all_weights_and_does_not_download_python_code():
    for p in PROFILES.values():
        manifest=json.loads(Path(p['manifest']).read_text())
        assert manifest['repo']==p['repo'] and manifest['revision']==p['revision'] and manifest['license']=='apache-2.0'
        weights=[f for f in manifest['files'] if f['rfilename'].endswith('.safetensors')]
        assert weights and all(len(f['lfs']['sha256'])==64 for f in weights)
        assert not any(f['rfilename'].endswith(('.py','.pkl','.pickle','.bin')) for f in manifest['files'])
