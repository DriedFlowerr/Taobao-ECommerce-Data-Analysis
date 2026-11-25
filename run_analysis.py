"""Entry point: python run_analysis.py [--stage data|funnel|inference|model|all]."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

from src.data import (audit_source, build_snapshots, connect_source, load_snapshots,
                      summarize_snapshots)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'


def save_json(data: dict, name: str) -> None:
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def main(stage='all', memory='600MB', threads=2):
    OUT.mkdir(exist_ok=True)
    if stage in ('all', 'data', 'funnel'):
        con = connect_source(ROOT, OUT, memory, threads)
        try:
            if stage in ('all', 'data'):
                print('1/4: reconciling raw and cleaned records...', flush=True)
                print(audit_source(con, ROOT, OUT), flush=True)
                print('Extracting point-in-time snapshots...', flush=True)
                path = build_snapshots(con, OUT)
            if stage in ('all', 'funnel'):
                from src.funnel import analyze_funnel
                print('2/4: computing activity and strict sequence funnel...', flush=True)
                result = analyze_funnel(con, OUT)
                save_json(result, 'funnel_summary.json')
                print(result, flush=True)
        finally:
            con.close()

    if stage in ('all', 'data', 'inference', 'model'):
        path = OUT / 'cache' / 'snapshots.parquet'
        if not path.exists():
            raise FileNotFoundError('Run python run_analysis.py --stage data first.')
        snapshots = load_snapshots(path)
        if stage in ('all', 'data'):
            summary = summarize_snapshots(snapshots, OUT)
            save_json(summary, 'snapshot_summary.json')
            print(summary, flush=True)
        if stage in ('all', 'inference'):
            from src.inference import analyze_association
            print('3/4: estimating held-out user-level associations...', flush=True)
            result = analyze_association(snapshots, OUT)
            save_json(result, 'association_summary.json')
            print('Association estimates:', result['crude'], result['standardized'], flush=True)
        if stage in ('all', 'model'):
            from src.modeling import train_and_evaluate
            print('4/4: training temporal benchmarks and evaluating fixed-budget policies...', flush=True)
            result = train_and_evaluate(snapshots, OUT)
            save_json(result, 'model_summary.json')
            from src.experiment import plan_trial
            import pandas as pd
            plan_trial(pd.DataFrame(result['targeting']), OUT)
            print('Selected model:', result['selected_model'], flush=True)
    print(f'Completed stage: {stage}. Outputs: {OUT}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['all', 'data', 'funnel', 'inference', 'model'], default='all')
    p.add_argument('--memory', default='600MB', help='DuckDB buffer memory limit')
    p.add_argument('--threads', type=int, default=2, help='DuckDB worker threads')
    args = p.parse_args()
    main(args.stage, args.memory, args.threads)
