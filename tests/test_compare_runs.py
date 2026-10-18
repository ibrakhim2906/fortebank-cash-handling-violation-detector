"""Run comparison reads evaluator reports and never relaxes promotion."""
import json
from scripts import compare_runs as c


def report(path,correct,localized,n=30,labelled=16,**extra):
    value=dict(n_videos=n,accuracy=correct/n,tp=12,fn=4,fp=3,failed=0,labelled_events=labelled,hit_2s=localized/labelled,
               type_accuracy=1.,worst_sec=70.,per_type={'food_drink':dict(labelled=2,localized=localized%3,typed=0)},**extra)
    path.write_text(json.dumps(value));return path


def test_promotion_needs_more_verdicts_and_no_lost_localization(tmp_path):
    base=c.summary(report(tmp_path/'baseline.json',23,5))
    assert c.promotable(c.summary(report(tmp_path/'a.json',25,5)),base)
    assert not c.promotable(c.summary(report(tmp_path/'b.json',25,4)),base)
    assert not c.promotable(c.summary(report(tmp_path/'c.json',23,9)),base)
    assert not c.promotable(c.summary(report(tmp_path/'d.json',26,9,n=29)),base)


def test_table_lists_every_run_and_type(tmp_path,capsys):
    paths=[report(tmp_path/'baseline.json',23,5),report(tmp_path/'recall.json',26,8)]
    c.main([str(p) for p in paths]+['--out',str(tmp_path/'out.md')])
    text=(tmp_path/'out.md').read_text()
    assert '| baseline | 23/30 |' in text and '| recall | 26/30 |' in text and '| yes |' in text
    assert '| food_drink |' in text and capsys.readouterr().out==text
