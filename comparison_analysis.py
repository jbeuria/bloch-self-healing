"""Statistics for compare.py outputs; reads saved data, never runs simulations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from .storage import write_csv, write_json, snapshot_analysis


def wilson_interval(successes, n, z=1.959963984540054):
    p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0., float(center-half)), min(1., float(center+half))


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return adjusted


def paired_inference(full, control, family_size, rng, bootstrap=20000, permutations=50000):
    full, control = np.asarray(full), np.asarray(control)
    if len(full) != len(control) or len(full) < 2:
        raise ValueError('paired inference requires >= 2 complete seed pairs')
    d = full-control
    indices = rng.integers(0, len(d), (bootstrap, len(d)))
    means = d[indices].mean(axis=1)
    fmeans, cmeans = full[indices].mean(axis=1), control[indices].mean(axis=1)
    improvement = 100*(1-fmeans/cmeans) if np.all(cmeans > 0) else np.full(bootstrap, np.nan)
    low, high = np.quantile(means, [.025, .975])
    alpha = .05/family_size
    family_low, family_high = np.quantile(means, [alpha/2, 1-alpha/2])
    if np.isfinite(improvement).all():
        relative_low, relative_high = np.quantile(improvement, [alpha/2, 1-alpha/2])
    else:
        relative_low = relative_high = None
    exceed = 0
    for start in range(0, permutations, 2000):
        signs = 2*rng.integers(0, 2, (min(2000, permutations-start), len(d)))-1
        exceed += np.count_nonzero(abs((signs*d).mean(axis=1)) >= abs(d.mean())-1e-14)
    return {
        'n': len(d), 'full_mean': float(full.mean()), 'control_mean': float(control.mean()),
        'difference': float(d.mean()), 'ci95_low': float(low), 'ci95_high': float(high),
        'family_ci_low': float(family_low), 'family_ci_high': float(family_high),
        'improvement_pct': float(100*(1-full.mean()/control.mean())) if control.mean() > 0 else None,
        'improvement_family_low': float(relative_low) if relative_low is not None else None,
        'improvement_family_high': float(relative_high) if relative_high is not None else None,
        'full_win_fraction': float(np.mean(d < -1e-12)), 'tie_fraction': float(np.mean(abs(d) <= 1e-12)),
        'permutation_p': float((1+exceed)/(permutations+1)),
    }


def load_table(directory):
    directory = Path(directory)
    protocol = json.loads((directory/'protocol.json').read_text())
    expected_hash = (directory/'protocol.sha256').read_text().strip()
    if hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest() != expected_hash:
        raise ValueError('protocol changed after freezing')
    completed = json.loads((directory/'complete.json').read_text())
    rows = list(csv.DictReader((directory/'summary.csv').open()))
    if completed['runs'] != len(rows) or completed['protocol_sha256'] != expected_hash:
        raise ValueError('completion manifest inconsistent')
    boolean = {'safe', 'progress_ok', 'obstacles_cleared', 'healing_success',
               'had_split', 'eventually_recovered', 'first_event_observed'}
    strings = {'case', 'scenario', 'controller', 'status'}
    for r in rows:
        for key, value in list(r.items()):
            if key in boolean:
                if value not in ('True', 'False'):
                    raise ValueError(f'invalid boolean {key}')
                r[key] = value == 'True'
            elif key not in strings:
                r[key] = float(value) if value else None
        r['seed'] = int(r['seed'])
    lookup = {(r['case'], r['controller'], r['seed']): r for r in rows}
    expected = {(case, mode, seed) for case in protocol['configs']
                for mode in protocol['controllers']
                for seed in (protocol['seeds'] if case in protocol['main_cases'] else protocol['control_seeds'])}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError('missing, duplicate or unexpected seed/controller cases')
    return rows, lookup, protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory')
    args = p.parse_args()
    out = Path(args.directory)
    rows, lookup, protocol = load_table(out)
    snapshot_analysis(out)
    rng = np.random.default_rng(371288)
    aggregate, contrasts = [], []
    for case in protocol['configs']:
        seeds = protocol['seeds'] if case in protocol['main_cases'] else protocol['control_seeds']
        endpoint = protocol['primary_endpoints'].get(case, 'fragmentation_burden')
        for mode in protocol['controllers']:
            group = [lookup[case, mode, seed] for seed in seeds]
            values = np.array([r[endpoint] for r in group])
            recovery = [r['first_recovery_time'] for r in group if r['first_event_observed']]
            healed = sum(r['healing_success'] for r in group)
            healing_ci = wilson_interval(healed, len(group))
            aggregate.append({
                'case': case, 'controller': mode, 'n': len(group), 'endpoint': endpoint,
                'mean': float(values.mean()), 'sd': float(values.std(ddof=1)) if len(group) > 1 else 0,
                'median': float(np.median(values)), 'q10': float(np.quantile(values,.1)),
                'q90': float(np.quantile(values,.9)),
                'split_runs': sum(r['had_split'] for r in group),
                'healed_runs': healed,
                'healing_rate_ci95_low': healing_ci[0],
                'healing_rate_ci95_high': healing_ci[1],
                'safe_runs': sum(r['safe'] for r in group),
                'progress_ok_runs': sum(r['progress_ok'] for r in group),
                'cleared_runs': sum(r['obstacles_cleared'] for r in group),
                'progress_mean': float(np.mean([r['progress_fraction'] for r in group])),
                'first_recovery_observed_n': len(recovery),
                'first_recovery_observed_mean_descriptive_only': float(np.mean(recovery)) if recovery else None,
                'minimum_cluster_mean': float(np.mean([r['min_cluster'] for r in group])),
            })
        if case not in protocol['main_cases']:
            continue
        full = [lookup[case, 'full', seed] for seed in seeds]
        for mode in protocol['controllers']:
            if mode == 'full':
                continue
            control = [lookup[case, mode, seed] for seed in seeds]
            result = paired_inference([r[endpoint] for r in full], [r[endpoint] for r in control],
                                      24, rng)
            progress_d = np.array([a['progress_fraction']-b['progress_fraction'] for a,b in zip(full,control)])
            indices = rng.integers(0, len(seeds), (20000, len(seeds)))
            progress_ci = np.quantile(progress_d[indices].mean(axis=1), [.025, .975])
            healing_d = np.array([int(a['healing_success'])-int(b['healing_success']) for a,b in zip(full,control)])
            healing_boot = healing_d[indices].mean(axis=1)
            healing_ci = np.quantile(healing_boot, [.025, .975])
            healing_family = np.quantile(healing_boot, [.05/48, 1-.05/48])
            rescued, worsened = int(np.sum(healing_d > 0)), int(np.sum(healing_d < 0))
            success_p = float(binomtest(rescued, rescued+worsened, .5).pvalue) if rescued+worsened else 1.
            contrasts.append({'case': case, 'control': mode, 'endpoint': endpoint, **result,
                              'progress_difference': float(progress_d.mean()),
                              'progress_ci95_low': float(progress_ci[0]),
                              'progress_ci95_high': float(progress_ci[1]),
                              'healing_success_difference': float(healing_d.mean()),
                              'healing_ci95_low': float(healing_ci[0]), 'healing_ci95_high': float(healing_ci[1]),
                              'healing_family_low': float(healing_family[0]), 'healing_family_high': float(healing_family[1]),
                              'rescued_seeds': rescued, 'worsened_seeds': worsened,
                              'healing_exact_mcnemar_p': success_p,
                              'marked_endpoint_supported': result['improvement_family_low'] is not None and result['improvement_family_low'] >= 20,
                              'progress_noninferior_exploratory': float(progress_ci[0]) >= -.02})
    for row, p_adjusted in zip(contrasts, holm([r['permutation_p'] for r in contrasts])):
        row['holm_p'] = float(p_adjusted)
    for row, p_adjusted in zip(contrasts, holm([r['healing_exact_mcnemar_p'] for r in contrasts])):
        row['healing_holm_p'] = float(p_adjusted)
    write_csv(out/'comparison_aggregate.csv', aggregate)
    write_csv(out/'comparison_contrasts.csv', contrasts)
    write_json(out/'comparison_analysis.json', {
        'bootstrap_resamples': 20000, 'sign_permutations': 50000, 'rng_seed': 371288,
        'primary_family_size': 24,
        'intervals': 'paired percentile bootstrap, ordinary 95% and Bonferroni-adjusted approximate 95% family coverage',
        'pvalues': 'two-sided paired sign permutations, Holm family correction; assumes paired differences exchangeable under sign under null',
        'sampling_unit': 'one independently generated seed, with pairing preserved between controllers',
        'scope': 'Monte Carlo variability conditional on fixed geometry/configuration; seeds are not independent environments',
        'progress_intervals': 'secondary unadjusted 95%, not a confirmatory noninferiority test',
        'success_inference': 'exploratory secondary 24-contrast family; paired bootstrap and exact McNemar with separate Holm correction; added after descriptive success counts were observed',
        'caveat': 'a fixed-parameter mechanism audit, not an independently tuned algorithm competition',
        'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    print('case control improvement% family95% progress_delta Holm_p')
    for r in contrasts:
        print(f"{r['case']:7} {r['control']:16} {r['improvement_pct']:7.2f} "
              f"[{r['improvement_family_low']:7.2f}, {r['improvement_family_high']:7.2f}] "
              f"{r['progress_difference']:+.4f} {r['holm_p']:.4g}")


if __name__ == '__main__':
    main()
