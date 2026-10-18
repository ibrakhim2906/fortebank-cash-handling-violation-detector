"""Compare frozen-evaluator reports against the first (baseline) report."""
import argparse
import json
from pathlib import Path


def summary(path):
    value=json.loads(Path(path).read_text())
    n=value['n_videos'];labelled=value['labelled_events']
    return dict(name=Path(path).stem,n=n,correct=round(value['accuracy']*n),tp=value['tp'],fn=value['fn'],fp=value['fp'],failed=value['failed'],
                labelled=labelled,localized=round(value['hit_2s']*labelled),type_accuracy=value.get('type_accuracy',0),
                worst_sec=value.get('worst_sec'),per_type=value.get('per_type',{}))


def promotable(row,base):
    """Promotion needs more correct verdicts without losing localized events."""
    return row['n']==base['n'] and row['labelled']==base['labelled'] and row['correct']>base['correct'] and row['localized']>=base['localized']


def table(rows):
    base=rows[0];lines=['| Run | Correct | TP | FN | FP | Failed | Localized ±2s | Type acc. | Worst s | Beats baseline |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        worst='' if row['worst_sec'] is None else f"{row['worst_sec']:.1f}"
        verdict='baseline' if row is base else 'yes' if promotable(row,base) else 'no'
        lines.append(f"| {row['name']} | {row['correct']}/{row['n']} | {row['tp']} | {row['fn']} | {row['fp']} | {row['failed']} | {row['localized']}/{row['labelled']} | {row['type_accuracy']:.2f} | {worst} | {verdict} |")
    kinds=sorted({k for row in rows for k in row['per_type']})
    if kinds:
        lines+=['','| Type (localized/labelled) | '+' | '.join(row['name'] for row in rows)+' |','|---|'+'---:|'*len(rows)]
        for kind in kinds:
            cells=[f"{row['per_type'].get(kind,{}).get('localized',0)}/{row['per_type'].get(kind,{}).get('labelled',0)}" for row in rows]
            lines.append(f'| {kind} | '+' | '.join(cells)+' |')
    return '\n'.join(lines)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reports',nargs='+',type=Path);p.add_argument('--out',type=Path)
    args=p.parse_args(argv);text=table([summary(path) for path in args.reports])
    if args.out:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
