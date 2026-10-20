"""Read generated primary runs, reconstruct endpoints and vary recovery thresholds."""
from pathlib import Path
from dataclasses import replace
import hashlib
import numpy as np
from .config import Config
from .metrics import detect_events
from .healing_refinement_analysis import load
from .storage import load_result, write_csv
from .paths import ROOT

DATA = ROOT / 'results/primary'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(case, mode, seed):
    arrays, _ = load_result(DATA / case / 'runs' / f'split__{mode}__seed{seed}.npz')
    return arrays


def recovered(d, c, r):
    events = detect_events(d['time'], d['cluster'], d['polar'], c)
    final = d['time'] >= d['time'][-1] - c.hold_time - 1e-9
    healed = bool(events and all(e['observed'] for e in events)
                  and np.all(d['cluster'][final] >= c.recovery_fraction)
                  and np.all(d['polar'][final] >= c.recovery_polar)
                  and r['safe'] and r['progress_ok'] and r['obstacles_cleared'])
    restricted = events[0]['duration'] if events and events[0]['observed'] else c.duration
    return events, healed, restricted


def thresholds(directory):
    global DATA
    DATA = Path(directory)
    _, lookup, protocol = load(DATA)
    groups, paired = {}, []
    if len(protocol['seeds']) < 2:
        raise ValueError('Threshold means and sample SD require at least two seeds')
    for case in protocol['main_cases']:
        for seed in protocol['seeds']:
            for mode in protocol['controllers']:
                d = load_run(case, mode, seed)
                c = Config(**protocol['selected'][mode]['config'])
                for fraction in (.9, .95, 1.):
                    for hold in (2., 5.):
                        _, healed, restricted = recovered(
                            d, replace(c, recovery_fraction=fraction, hold_time=hold),
                            lookup[case, mode, seed])
                        row = dict(case=case, controller=mode, seed=seed,
                                   recovery_fraction=fraction, hold_time=hold,
                                   healed=healed, restricted_recovery_time=restricted)
                        paired.append(row)
                        groups.setdefault((case, mode, fraction, hold), []).append(row)
    result = []
    for (case, mode, fraction, hold), rows in groups.items():
        values = [r['restricted_recovery_time'] for r in rows]
        result.append(dict(case=case, controller=mode, recovery_fraction=fraction,
                           hold_time=hold, n=len(rows), healed_runs=sum(r['healed'] for r in rows),
                           mean_restricted_recovery_time=float(np.mean(values)),
                           sd_restricted_recovery_time=float(np.std(values, ddof=1))))
    write_csv(DATA / 'threshold_sensitivity.csv', result)
    write_csv(DATA / 'threshold_sensitivity_paired.csv', paired)
    print(f'Reconstructed {len(paired)} trial/threshold outcomes and {len(result)} aggregate rows.')
