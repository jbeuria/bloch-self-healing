"""Predeclared paired-seed mechanism audit; no tuning on evaluation seeds.

This compares removals/replacements at shared physical parameters. It is not a
claim that independently optimized competing algorithms have been beaten.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .config import Config
from .controller import stationary_z
from .simulation import simulate
from .storage import prepare_output, save_result, write_csv, write_json

CONTROLLERS = ('full', 'fast', 'frozen', 'scalar', 'reactive', 'scalar_matched', 'instant_matched')
MAIN_CASES = ('split', 'single', 'double', 'triple')


def equilibrium_tone(c):
    """Homogeneous aligned stationary self state; no simulation outcome used."""
    s = c.slow_base
    for _ in range(10000):
        next_s = c.slow_base + float(np.tanh(c.feedback_gain*stationary_z(
            c.g_align+c.kappa*s, c.gamma, c.t1, c.t2)))
        if abs(next_s-s) < 1e-13:
            return next_s
        s = next_s
    raise ValueError('stationary self iteration did not converge')


def endpoints(result):
    r, c = result.summary, result.config
    t, f = result.series['time'], result.series['cluster']
    burden = float(np.sum((2-f[:-1]-f[1:])*np.diff(t)/2))
    # Uniform observation horizon applies to the imposed split only.
    recovery = (float(r['first_recovery_time']) if r['first_event_observed']
                else c['duration']) if r['scenario'] == 'split' else None
    return {'restricted_recovery_time': recovery, 'fragmentation_burden': burden,
            'had_split': bool(r['split_events']),
            'eventually_recovered': bool(r['split_events'] and r['recovered_events'] == r['split_events'])}


def run_one(job):
    config, case, seed, mode, out, record = job
    result = simulate(Config(**config), 'split' if case == 'no_centering' else case,
                      seed, mode, record=record)
    save_result(result, Path(out)/case/'runs')
    return {'case': case, **result.summary, **endpoints(result)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='self_healing/configs/tuned_full.json')
    p.add_argument('--seeds', type=int, default=100)
    p.add_argument('--first-seed', type=int, default=10001)
    p.add_argument('--control-seeds', type=int, default=10)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if min(args.seeds, args.control_seeds, args.workers) < 1 or args.first_seed < 0:
        p.error('counts must be positive; first seed nonnegative')
    base = Config.load(args.config)
    # Constant bias is determined by homogeneous steady state, not arbitrary .75.
    base = replace(base, frozen_tone=equilibrium_tone(base))
    cases = {case: base for case in MAIN_CASES}
    cases.update(open=base, no_centering=replace(base, center_gain=0))
    seeds = list(range(args.first_seed, args.first_seed+args.seeds))
    control_seeds = list(range(args.first_seed+args.seeds,
                               args.first_seed+args.seeds+args.control_seeds))
    out = prepare_output(args.out)
    protocol = {
        'design': 'paired seeds, fixed shared physical parameters, mechanism ablations',
        'config_source': str(Path(args.config).resolve()),
        'controllers': CONTROLLERS, 'main_cases': MAIN_CASES,
        'seeds': seeds, 'control_seeds': control_seeds,
        'configs': {k: c.to_dict() for k, c in cases.items()},
        'primary_endpoints': {'split': 'restricted_recovery_time',
                             'single': 'fragmentation_burden', 'double': 'fragmentation_burden',
                             'triple': 'fragmentation_burden'},
        'primary_family': 'full minus each of six controls, across four main cases (24 contrasts)',
        'marked_benefit_target': 'at least 20% endpoint reduction; report adjusted intervals; progress loss <= .02',
        'recovery_rule': 'F >= .9 and P >= .9 continuously for 2 time units; all failures retained',
        'no_split_rule': 'prevention, never healing; natural-case burden includes all runs',
        'censoring_rule': 'imposed split unrecovered by horizon assigned horizon; natural cases use whole-run burden',
        'statistics': 'seed-paired percentile bootstrap, 20000 resamples; Bonferroni family intervals; paired sign permutation with Holm correction',
        'inference_scope': 'specified initialization/noise distribution, fixed scenario geometries and shared parameters',
        'selection': 'existing pilot-tuned full configuration frozen before these seeds; no evaluation tuning',
        'recording': 'all per-step metrics saved; spatial snapshots for first 3 seeds per case, chosen before simulation',
        'workers': args.workers,
    }
    write_json(out/'protocol.json', protocol)
    (out/'protocol.sha256').write_text(hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()+'\n')
    jobs = [(c.to_dict(), case, seed, mode, str(out), seed in selected[:3])
            for case, c in cases.items()
            for selected in [seeds if case in MAIN_CASES else control_seeds]
            for seed in selected for mode in CONTROLLERS]
    rows = []
    # Set before spawned worker imports numerical libraries; avoid nested BLAS pools.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    print(f'Frozen protocol: {out}/protocol.json; {len(jobs)} runs', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_one, jobs, chunksize=1):
            rows.append(row)
            if len(rows) % 35 == 0 or len(rows) == len(jobs):
                write_csv(out/'summary.csv', rows)
                print(f'{len(rows)}/{len(jobs)} completed; {row["case"]} seed {row["seed"]}', flush=True)
    write_json(out/'complete.json', {'runs': len(rows), 'protocol_sha256': hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()})


if __name__ == '__main__':
    main()
