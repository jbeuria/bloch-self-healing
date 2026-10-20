#!/usr/bin/env python3
"""Run and verify the fixed-parameter manuscript experiments from source only."""
import argparse
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from bootstrap import activate
ROOT = activate()
SELECTION = ROOT / 'configurations/working.json'


@contextmanager
def arguments(values):
    previous = sys.argv
    sys.argv = [str(x) for x in values]
    try:
        yield
    finally:
        sys.argv = previous


def check():
    manifest = ROOT / 'MANIFEST.sha256'
    count = 0
    for line in manifest.read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'File missing or changed: {name}')
        count += 1
    from self_healing.config import Config
    assert hashlib.sha256(SELECTION.read_bytes()).hexdigest() == SELECTION.with_suffix('.sha256').read_text().strip()
    for spec in json.loads(SELECTION.read_text()).values():
        Config(**spec['config']).validate()
    print(f'Integrity check: {count} source/configuration files match the release manifest.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'tests', 'smoke', 'evaluate',
                        'verify-run', 'stats', 'thresholds', 'generalization', 'figures'])
    parser.add_argument('--out', type=Path, help='Output folder, or completed primary run for verify-run/stats/thresholds')
    parser.add_argument('--primary', type=Path, default=ROOT/'results/primary', help='Completed primary run used for figures')
    parser.add_argument('--sensitivity', type=Path, default=ROOT/'results/generalization', help='Completed sensitivity run used for figures')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seeds', type=int)
    parser.add_argument('--first-seed', type=int)
    parser.add_argument('--with-c-oracle', action='store_true', help='Include optional tests that compile a pinned upstream C oracle')
    args = parser.parse_args()
    if args.workers < 1 or (args.seeds is not None and args.seeds < 1):
        parser.error('workers and seeds must be positive')
    if args.first_seed is not None and args.first_seed < 1:
        parser.error('first-seed must be positive')
    if args.stage == 'generalization' and args.seeds == 1:
        parser.error('generalization analysis requires at least two seeds')
    def output(default):
        return (args.out if args.out is not None else ROOT/'results'/default).resolve()
    if args.stage == 'check':
        check()
    elif args.stage == 'tests':
        import pytest
        options = [str(ROOT/'tests'), '-q', '-p', 'no:cacheprovider']
        if not args.with_c_oracle:
            options += ['-k', 'not test_vasarhelyi_matches_authors_c_interaction_bodies and not test_braking_curve_against_upstream_and_damping_energy']
        raise SystemExit(pytest.main(options))
    elif args.stage in ('smoke', 'evaluate'):
        import run_primary
        directory = output('smoke' if args.stage == 'smoke' else 'primary')
        seeds = args.seeds if args.seeds is not None else (1 if args.stage == 'smoke' else 100)
        with arguments(['run_primary.py', '--out', directory, '--workers', args.workers,
                        '--seeds', seeds, '--first-seed', args.first_seed or 73001]):
            run_primary.main()
        from self_healing.verify import primary
        primary(directory)
    elif args.stage == 'verify-run':
        from self_healing.verify import primary
        primary(output('primary'))
    elif args.stage == 'stats':
        from self_healing.verify import primary
        from self_healing.healing_refinement_analysis import load, main as analyze
        directory = output('primary')
        _, _, protocol = load(directory)
        if len(protocol['seeds']) < 2:
            parser.error('Means with sample SD and paired inference require at least two seeds')
        primary(directory)
        with arguments(['primary-analysis', directory]), redirect_stdout(io.StringIO()):
            analyze()
        report = directory/'RESULTS.txt'
        text = report.read_text().replace('OPTIMIZED RECOVERY: FROZEN HELD-OUT RESULTS', 'FIXED-PARAMETER RECOVERY RESULTS')
        text = text.replace('Optimized full', 'Full').replace('optimized full', 'full')
        text = text.replace('Original-full comparisons measure parameter-optimization gains, not self-feedback causality.\n', '')
        text = text.replace('The three external algorithms were independently tuned with identical final evaluation budgets: 384 training + 72 validation runs each, as was full.',
                            'The present evaluation uses fixed working coefficients. Earlier parameter development is described in the supplement.')
        report.write_text(text)
        print(f'Primary means, sample SD and paired statistical comparisons written to {directory}')
    elif args.stage == 'thresholds':
        from self_healing.verify import primary
        from self_healing.bundle import thresholds
        directory = output('primary')
        primary(directory)
        thresholds(directory)
    elif args.stage == 'generalization':
        from self_healing.generalization import evaluate
        from self_healing.generalization_analysis import audit, analyze
        directory = output('generalization')
        evaluate(directory, workers=args.workers, seeds=args.seeds or 30,
                 first_seed=args.first_seed or 74001, selection=SELECTION)
        audit(directory, directory)
        analyze(directory, directory)
    else:
        from self_healing.verify import primary
        from self_healing.generalization_analysis import read
        _, _, protocol = read(args.sensitivity)
        if protocol['selection_sha256'] != hashlib.sha256(SELECTION.read_bytes()).hexdigest():
            raise ValueError('Sensitivity run did not use the manuscript configuration')
        primary(args.primary)
        directory = output('assets')
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.update(HEALING_DATA=str(args.primary.resolve()), HEALING_SELECTION=str(SELECTION),
                          HEALING_GENERALIZATION=str(args.sensitivity.resolve()), HEALING_ASSET_OUT=str(directory),
                          MPLCONFIGDIR=str(ROOT/'results/.matplotlib'))
        for name in ('build_assets.py', 'build_generalization.py', 'build_compact_tables.py'):
            runpy.run_path(str(ROOT/'plotting'/name), run_name='__main__')


if __name__ == '__main__':
    main()
