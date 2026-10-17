"""Conservative engineering rubric proxy; never an official jury grade."""
import argparse,json,math
from pathlib import Path


def calculate(honest, *, augmented=None, api=None, gates=None, legal_share=0):
    gates=gates or {}; augmented=augmented or {}; api=api or {}
    # A resubstitution or stub report cannot acquire the 'honest' label by CLI.
    honest_valid=honest.get('validation_kind')=='leave-one-setup-out' and honest.get('fold_integrity_verified') is True and honest.get('mode') in ['full','api','cached']
    def fraction(v):
        return float(v) if type(v) in [int,float] and math.isfinite(v) and 0<=v<=1 else 0
    accuracy=fraction(honest.get('accuracy',0)) if honest_valid else 0
    count=api.get('n_videos');latency=api.get('worst_sec')
    full_api=api.get('mode')=='api' and api.get('subset')=='all' and type(count) is int and count>=40 and api.get('completed')==count and api.get('failed')==0
    product=full_api and gates.get('pytest_green') is True and type(latency) in [int,float] and math.isfinite(latency) and 0<=latency<=200
    tech=gates.get('network_blocked_processing_passed') is True and gates.get('license_table_verified') is True
    robustness=0
    clean_accuracy=fraction(augmented.get('clean_accuracy',accuracy))
    if honest_valid and augmented.get('all_requested_transforms_complete') is True and clean_accuracy>0:
        robustness=10*min(1,fraction(augmented.get('accuracy',0))/clean_accuracy)*accuracy
    parts=dict(detection=25*accuracy,
               localization=15*(.5*fraction(honest.get('hit_1s',0))+.5*fraction(honest.get('hit_2s',0))) if honest_valid else 0,
               classification=15*fraction(honest.get('type_accuracy',0)) if honest_valid else 0,
               legal=15*fraction(legal_share) if honest_valid else 0,
               robustness=robustness,product=10*bool(product),tech=10*bool(tech))
    targets=dict(score_at_least_90=sum(parts.values())>=90,recall_1=honest_valid and honest.get('recall')==1,
                 false_alarms_at_most_1=honest_valid and honest.get('fp',10**9)<=1,
                 timecodes_90_percent=honest_valid and honest.get('hit_2s',0)>=.9,
                 type_85_percent=honest_valid and honest.get('type_accuracy',0)>=.85,
                 repeated_worst_at_most_200=gates.get('two_worst_case_runs_passed') is True)
    return dict(estimated_proxy_score=sum(parts.values()),components=parts,targets=targets,
                honest_validation_eligible=honest_valid,official_jury_score=False,
                disclosure=honest.get('disclosure','No independence evidence supplied'),
                missing_gates=[k for k in ['pytest_green','network_blocked_processing_passed','license_table_verified','two_worst_case_runs_passed'] if gates.get(k) is not True])


def legal_fraction(report,rules):
    from scripts.eval_video import type_key
    types=rules.get('types',{}); total=points=0
    for record in report.get('records',[]):
        if record.get('status')!='done':continue
        for event in record.get('result',{}).get('violations',[]):
            total+=1; key=type_key(event.get('type',''),types)
            if key is None:continue
            sources=types[key].get('rules',[])
            if not sources:points+=.5
            elif any(s.get('verified') is True and s.get('source_url') and s.get('point') and s.get('quote') for s in sources):points+=1
    return points/total if total else 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--honest',type=Path,required=True)
    for name in ['holdout','augmented','api','gates','rules']:p.add_argument('--'+name,type=Path)
    p.add_argument('--json',type=Path);args=p.parse_args(argv)
    read=lambda f:json.loads(f.read_text()) if f else None
    honest=read(args.honest);rules=read(args.rules) or {}
    value=calculate(honest,augmented=read(args.augmented),api=read(args.api),gates=read(args.gates),legal_share=legal_fraction(honest,rules))
    if args.holdout:
        reserve=read(args.holdout);value['reserve']={k:reserve.get(k) for k in ['accuracy','recall','fp','hit_2s','type_accuracy','disclosure']}
    payload=json.dumps(value,indent=2,ensure_ascii=False)
    if args.json:args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(payload+'\n')
    print(payload)


if __name__=='__main__':main()
