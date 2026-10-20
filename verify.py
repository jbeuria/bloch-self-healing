"""Independent endpoint checks on generated runs; requires no reference dataset."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from .config import Config
from .compare import equilibrium_tone
from .healing_refinement_analysis import load
from .storage import load_result, write_json
from .bundle import recovered, digest
from .paths import ROOT


def primary(directory):
    root = Path(directory)
    rows, lookup, protocol = load(root)
    if protocol['selection_sha256'] != digest(ROOT / 'configurations/working.json'):
        raise ValueError('Run did not use the manuscript working configuration')
    selected = json.loads((ROOT / 'configurations/working.json').read_text())
    close = lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)
    hashes = {}
    initial = {}
    for (case, mode, seed), row in lookup.items():
        spec = selected[mode if mode in selected else 'full']
        assert protocol['selected'][mode] == spec
        path = root / case / 'runs' / f'split__{mode}__seed{seed}.npz'
        d, meta = load_result(path)
        c = replace(Config(**spec['config']), noise=protocol['tasks'][case]['noise'])
        c = replace(c, frozen_tone=equilibrium_tone(c))
        for field, value in c.to_dict().items():
            assert meta['config'][field] == value, (case, mode, seed, field)
        assert meta['config']['controller_parameters'] == spec['parameters']
        t, f, p = (d[k] for k in ('time', 'cluster', 'polar'))
        assert len(t) == round(c.duration / c.dt) + 1
        close(t, np.arange(len(t)) * c.dt)
        assert np.isfinite(f).all() and np.isfinite(p).all()
        assert f.min() > 0 and f.max() <= 1 and p.min() >= 0 and p.max() <= 1 + 1e-12
        assert f[0] < c.split_fraction and row['had_split']
        events, healed, restricted = recovered(d, c, row)
        assert events == meta['events']
        assert len(events) == row['split_events']
        assert sum(e['observed'] for e in events) == row['recovered_events']
        assert healed == row['healing_success']
        close(restricted, row['restricted_recovery_time'])
        close(np.sum((2-f[:-1]*p[:-1]-f[1:]*p[1:])*np.diff(t)/2), row['joint_recovery_deficit'])
        close(np.sum((2-f[:-1]-f[1:])*np.diff(t)/2), row['fragmentation_burden'])
        close((d['centroid_y'][-1]-d['centroid_y'][0])/(c.speed*c.duration), row['progress_fraction'])
        assert row['progress_ok'] == (row['progress_fraction'] >= c.min_progress_fraction)
        assert row['safe'] == (row['min_pair_clearance'] >= -1e-10 and
                              (row['min_obstacle_clearance'] is None or row['min_obstacle_clearance'] >= -1e-10))
        assert d['mean_speed'].max() <= c.speed + 1e-10
        assert d['max_bloch_norm'].max() <= 1 + 1e-10
        assert np.max(abs(d['tone']-c.slow_base)) <= 1 + 1e-10
        assert d['command_acceleration_mean'].max() <= 2 + 1e-10
        for key, value in meta['summary'].items():
            if key in row:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    close(value, row[key])
                else:
                    assert value == row[key], (case, mode, seed, key)
        initial[case, mode, seed] = {
            'fp': (f[0], p[0]), 'internal': (d['tone'][0], d['max_bloch_norm'][0]),
            'positions': d['positions'][0] if len(d['positions']) else None,
            'headings': d['headings'][0] if len(d['headings']) else None}
        hashes[str(path.relative_to(root))] = digest(path)
    for (case, mode, seed), state in initial.items():
        full = initial[case, 'full', seed]
        close(state['fp'], full['fp'])
        if mode == 'frozen':
            close(state['internal'], full['internal'])
        for field in ('positions', 'headings'):
            if state[field] is not None:
                close(state[field], full[field])
    write_json(root/'raw_manifest.json', hashes)
    report = {'status': 'passed', 'runs_verified': len(rows),
              'selection_sha256': protocol['selection_sha256'],
              'protocol_sha256': digest(root/'protocol.json'),
              'checks': 'Coverage, fixed configuration, events, T/J, final recovery, progress, internal bounds, summary consistency, stored clearances, paired initial F/P and available spatial states',
              'limitation': 'This checks generated results internally. It does not compare against a bundled dataset or independently reconstruct every swept clearance. Initial internal matching uses recorded summaries; exact full-state copying is tested separately.'}
    write_json(root/'verification.json', report)
    print(f'Checked {len(rows)} generated primary trials: endpoints and protocol are consistent.')
