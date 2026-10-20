#!/usr/bin/env python3
"""Run the fixed manuscript configuration, without parameter search."""
from pathlib import Path
import argparse
import hashlib
import json
import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

from bootstrap import activate
activate()

from self_healing.config import Config
from self_healing.healing_refinement import TASKS, execute
from self_healing.external_controllers import PLANT
from self_healing.storage import prepare_output, write_json

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/primary'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seeds', type=int, default=100)
    parser.add_argument('--first-seed', type=int, default=73001)
    args = parser.parse_args()
    if args.workers < 1 or args.seeds < 1 or args.first_seed < 1:
        parser.error('workers, seeds and first-seed must be positive')
    source = ROOT / 'configurations/working.json'
    chosen = json.loads(source.read_text())
    selection_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if selection_hash != (source.parent / 'working.sha256').read_text().strip():
        raise ValueError('working configuration changed; restore it to reproduce the paper')
    for spec in chosen.values():
        Config(**spec['config']).validate()
    for mode in ('fast', 'frozen', 'scalar_matched'):
        chosen[mode] = chosen['full']
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    old = {41001, 41002, 43001, 43002, *range(44001, 44007),
           *range(51001, 51101), *range(62001, 62031)}
    if set(seeds) & old:
        raise ValueError('use seeds outside previous development/evaluation sets')
    out = prepare_output(ROOT / args.out)
    protocol = {
        'status': 'Fixed manuscript configuration; no search or retuning',
        'controllers': list(chosen), 'tasks': TASKS, 'main_cases': list(TASKS),
        'seeds': seeds, 'selected': chosen, 'plant': PLANT,
        'selection_sha256': selection_hash,
        'primary_endpoints': ['restricted_recovery_time', 'joint_recovery_deficit'],
        'scope': 'Fixed-parameter evaluation of the revised manuscript',
        'recording': 'First three seeds per task/controller, selected before simulation',
    }
    write_json(out / 'protocol.json', protocol)
    protocol_hash = hashlib.sha256((out / 'protocol.json').read_bytes()).hexdigest()
    (out / 'protocol.sha256').write_text(protocol_hash + '\n')
    jobs = [(spec, mode, case, seed, 0, str(out), seed in seeds[:3])
            for case in TASKS for seed in seeds for mode, spec in chosen.items()]
    rows = execute(jobs, out, args.workers, 'summary')
    write_json(out / 'complete.json', {
        'runs': len(rows), 'protocol_sha256': protocol_hash,
        'selection_sha256': selection_hash,
        'scope': 'Execution smoke check only' if args.seeds == 1 else 'Fixed-parameter evaluation',
    })


if __name__ == '__main__':
    main()
