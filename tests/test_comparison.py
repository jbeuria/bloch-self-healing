from dataclasses import replace
import numpy as np
from self_healing.config import Config
from self_healing.controller import stationary_z, Controller
from self_healing.compare import endpoints, equilibrium_tone
from self_healing.simulation import simulate


def test_stationary_response_solves_continuous_bloch_equations():
    hz = np.linspace(-4, 4, 30)
    gamma, t1, t2 = 1.3, .7, .4
    z = stationary_z(hz, gamma, t1, t2)
    y = -2*gamma*t2*z/(1+4*hz**2*t2**2)
    x = -2*hz*t2*y
    np.testing.assert_allclose(-2*hz*y-x/t2, 0, atol=1e-14)
    np.testing.assert_allclose(2*(hz*x-gamma*z)-y/t2, 0, atol=1e-14)
    np.testing.assert_allclose(2*gamma*y-(z-np.tanh(hz))/t1, 0, atol=1e-14)
    np.testing.assert_allclose(stationary_z(hz, 0, t1, t2), np.tanh(hz))


def test_frozen_tone_uses_stationary_self_consistency():
    c = Config()
    s = equilibrium_tone(c)
    expected = c.slow_base + np.tanh(c.feedback_gain*stationary_z(c.g_align+c.kappa*s, c.gamma, c.t1, c.t2))
    assert abs(expected-s) < 1e-12


def test_zero_feedback_removes_self_effect_exactly():
    c = replace(Config(), n=12, duration=3, kappa=0, caution_gain=0)
    full, fast = [simulate(c, 'split', 5, mode) for mode in ('full', 'fast')]
    np.testing.assert_array_equal(full.positions, fast.positions)
    assert full.series['tone'][-1] > 0


def test_matched_filters_keep_self_feedback_and_inactive_locality():
    c = replace(Config(), n=8)
    for mode in ('scalar_matched', 'instant_matched'):
        controller = Controller(c, mode)
        pos = np.column_stack((np.arange(8)*20., np.zeros(8)))
        controller.step(pos, np.zeros(8), np.empty((0,3)), np.zeros(8))
        np.testing.assert_array_equal(controller.s, 0)
        np.testing.assert_array_equal(controller.m, 0)


def test_failed_forced_recovery_kept_at_horizon():
    c = replace(Config(), n=12, duration=3, center_gain=0)
    result = simulate(c, 'split', 5, 'full')
    e = endpoints(result)
    assert e['restricted_recovery_time'] == c.duration
    assert e['fragmentation_burden'] > 0
    assert not e['eventually_recovered']


def test_paired_statistics_preserve_pairing_and_adjust_multiple_tests():
    from self_healing.comparison_analysis import paired_inference, holm
    control = np.arange(1., 41.)
    full = control*.5
    r = paired_inference(full, control, 24, np.random.default_rng(1),
                         bootstrap=1000, permutations=1000)
    assert abs(r['improvement_pct']-50) < 1e-12
    assert abs(r['improvement_family_low']-50) < 1e-12
    assert abs(r['improvement_family_high']-50) < 1e-12
    assert r['family_ci_high'] < 0
    same = paired_inference(control, control, 24, np.random.default_rng(2),
                            bootstrap=1000, permutations=1000)
    assert same['difference'] == 0 and same['permutation_p'] == 1
    np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
