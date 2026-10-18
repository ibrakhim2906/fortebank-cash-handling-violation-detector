from scripts.verify_license_policy import classify


def row(name='pkg',expression=None,classifiers=(),metadata=None):
    return dict(name=name,expression=expression,license_classifiers=list(classifiers),license_metadata=metadata)


def test_permissive_and_weak_copyleft_runtime_pass():
    assert classify(row(expression='MIT'),{})[0]=='permissive'
    assert classify(row(metadata='MPL-2.0 AND MIT'),{})[0]=='weak_copyleft'
    assert classify(row(expression='LGPL-2.1-only',classifiers=['License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)']),{})[0]=='weak_copyleft'
    assert classify(row(expression='PSF-2.0'),{})[0]=='psf'


def test_nvidia_terms_only_for_nvidia_runtime():
    assert classify(row('nvidia-cublas-cu12',metadata='NVIDIA Proprietary Software'),{})[0]=='nvidia_cuda'
    assert classify(row('vendor-sdk',metadata='NVIDIA Proprietary Software'),{})[0] is None


def test_fails_closed_on_gpl_noncommercial_and_unknown():
    assert classify(row(expression='GPL-3.0-only'),{})[0] is None
    assert classify(row(classifiers=['License :: OSI Approved :: GNU General Public License v3 (GPLv3)']),{})[0] is None
    assert classify(row(expression='AGPL-3.0'),{})[0] is None
    assert classify(row(metadata='CC-BY-NC-4.0'),{})[0] is None
    assert classify(row(metadata='UNKNOWN'),{})[0] is None


def test_notice_resolution_requires_the_preserved_text():
    unknown=row('xformers',metadata='UNKNOWN')
    assert classify(unknown,{})[0] is None
    assert classify(unknown,{'model/xformers-0.0.32.post1.dist-info/LICENSE':'BSD 3-Clause License'})[0]=='permissive'
