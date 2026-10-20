from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from self_healing.config import Config
from self_healing.controller import bloch_step, Controller
from self_healing.simulation import simulate


def test_bloch_ball_for_boundary_states_and_large_steps():
    rng = np.random.default_rng(43)
    m = rng.normal(size=(300, 3)); m /= np.linalg.norm(m, axis=1)[:, None]
    hz = rng.uniform(-8, 8, len(m))
    for dt in (.01, .05, .5, 2.0):
        updated = bloch_step(m, hz, 2, .6, 1.2, dt)
        assert np.max(np.linalg.norm(updated, axis=1)) <= 1+1e-12


def test_zero_transverse_drive_matches_scalar_exact_solution():
    hz = np.array([-.5, 0, 2.0])
    initial = np.zeros((3, 3)); initial[:, 2] = [.2, -.6, .8]
    actual = bloch_step(initial, hz, 0, .6, .3, .2)
    expected = np.tanh(hz)+(initial[:, 2]-np.tanh(hz))*np.exp(-.2/.6)
    np.testing.assert_allclose(actual[:, 2], expected, atol=1e-14)


def test_integrator_converges_to_continuous_equations():
    hz, gamma, t1, t2 = 1.7, 1.0, .6, .3
    def rhs(t, m):
        x,y,z = m
        return [-2*hz*y-x/t2, 2*(hz*x-gamma*z)-y/t2,
                2*gamma*y-(z-np.tanh(hz))/t1]
    initial = np.array([.1, .2, -.4])
    exact = solve_ivp(rhs, (0, 1), initial, rtol=1e-11, atol=1e-13).y[:, -1]
    errors = []
    for dt in (.1, .05, .025):
        m = initial.copy()
        for _ in range(round(1/dt)):
            m = bloch_step(m, np.array(hz), gamma, t1, t2, dt)
        errors.append(np.linalg.norm(m-exact))
    assert errors[1] < errors[0]/3
    assert errors[2] < errors[1]/3


def test_passive_is_exact_negative_control():
    c = replace(Config(), n=12, duration=3)
    fast = simulate(c, "split", 3, "fast")
    passive = simulate(c, "split", 3, "passive")
    np.testing.assert_array_equal(fast.positions, passive.positions)
    assert passive.series["tone"][-1] > 0
    assert fast.series["tone"][-1] == 0


def test_gamma_zero_full_and_scalar_have_same_trajectory():
    c = replace(Config(), n=12, gamma=0, duration=3)
    full = simulate(c, "split", 4, "full")
    scalar = simulate(c, "split", 4, "scalar")
    np.testing.assert_allclose(full.positions, scalar.positions, atol=1e-12)


@pytest.mark.parametrize("policy", ["reset", "retain", "decay"])
def test_inactive_channels_do_not_observe_disconnected_agents(policy):
    c = replace(Config(), n=8, channel_policy=policy)
    controller = Controller(c, "full")
    controller.m[:] = [.1, .2, .3]
    pos = np.column_stack([np.arange(c.n)*20, np.zeros(c.n)])
    controller.step(pos, np.zeros(c.n), np.empty((0,3)), np.zeros(c.n))
    expected = np.array([.1,.2,.3])
    if policy == "reset": expected *= 0
    if policy == "decay": expected *= np.exp(-c.dt/np.array([c.t2,c.t2,c.t1]))
    np.testing.assert_allclose(controller.m[0,1], expected)


def test_invalid_relaxation_and_unknown_parameters_rejected():
    with pytest.raises(ValueError): replace(Config(), t2=3).validate()
    with pytest.raises(ValueError): replace(Config(), dt=0).validate()
    with pytest.raises(ValueError): replace(Config(), recovery_fraction=.7).validate()


def test_local_controller_cannot_rejoin_remote_groups_without_shared_assistance():
    c = replace(Config(), n=12, center_gain=0, split_angle=0, noise=0,
                duration=5, initial_jitter=0, cohesion_gain=0)
    result = simulate(c, "split", 2, "reactive")
    assert not result.summary["healing_success"]
    assert not result.series["cross_edge"].any()


def test_bounded_turn_and_speed():
    c = replace(Config(), n=8)
    ctrl = Controller(c, "full")
    pos = np.column_stack([np.arange(8), np.zeros(8)]).astype(float)
    theta = np.full(8, -np.pi/2)
    new, angle = ctrl.step(pos, theta, np.empty((0,3)), np.zeros(8))
    assert np.max(abs(angle-theta)) <= c.max_turn_rate*c.dt+1e-12
    np.testing.assert_allclose(np.linalg.norm(new-pos,axis=1), c.speed*c.dt)
