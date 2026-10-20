from dataclasses import replace
import numpy as np
from self_healing.config import Config, SCENARIOS
from self_healing.scenarios import make_scenario
from self_healing.metrics import detect_events, observe, swept_clearances
from self_healing.simulation import simulate
from self_healing.storage import save_result, load_result
from self_healing.safety import safe_step_lengths


def test_imposed_split_is_disconnected_for_all_pilot_seeds():
    c=Config()
    for seed in range(10):
        pos, theta, groups, obs=make_scenario(c,"split",seed)
        row=observe(pos,theta,groups,obs,c)
        assert row["cluster"] == .5
        assert not row["cross_edge"]
        assert row["cross_gap"] >= c.split_gap


def test_layouts_fixed_across_seeds_and_have_expected_obstacle_counts():
    c=Config()
    for name,count in zip(SCENARIOS,[0,0,1,2,3]):
        a=make_scenario(c,name,1)[-1]
        b=make_scenario(c,name,2)[-1]
        assert len(a)==count
        np.testing.assert_array_equal(a,make_scenario(c,name,1)[-1])
        np.testing.assert_array_equal(a,b)


def test_recovery_uses_physical_hold_time_and_terminal_censoring():
    c=replace(Config(), hold_time=2)
    t=np.arange(8,dtype=float)
    events=detect_events(t,np.array([.5,.5,1,1,1,.5,.5,.5]),np.ones(8),c)
    assert events[0]["recovery_time"] == 2
    assert events[0]["confirmation_time"] == 4
    assert events[0]["duration"] == 2
    assert events[1]["duration"] == 2
    assert not events[1]["observed"]


def test_short_threshold_crossings_do_not_count_as_healing():
    c=replace(Config(),hold_time=2)
    events=detect_events(np.arange(5.),np.array([.5,1,.5,1,1]),np.ones(5),c)
    assert len(events)==1 and not events[0]["observed"]


def test_swept_safety_catches_crossing_between_frames():
    a=np.array([[-1.,0],[1,0]])
    b=-a
    pair,_=swept_clearances(a,b,np.empty((0,3)),.2)
    assert abs(pair+.4)<1e-12
    _,obstacle=swept_clearances(np.array([[-2.,0]]),np.array([[2.,0]]),np.array([[0,0,1]]),.2)
    assert abs(obstacle+1.2)<1e-12


def test_simulation_roundtrip_and_determinism(tmp_path):
    c=replace(Config(),n=12,duration=3)
    result=simulate(c,"split",7,"full")
    repeated=simulate(c,"split",7,"full")
    np.testing.assert_array_equal(result.positions,repeated.positions)
    save_result(result,tmp_path)
    data,metadata=load_result(tmp_path/"split__full__seed7.npz")
    np.testing.assert_array_equal(data["positions"],result.positions)
    assert metadata["config"]==c.to_dict()
    assert data["time"][-1]==c.duration




def test_local_braking_prevents_contact_for_arbitrary_headings():
    c=Config()
    before=np.array([[-.2,0],[.2,0]])
    lengths=safe_step_lengths(before,np.empty((0,3)),c)
    after=before+np.array([[1,0],[-1,0]])*lengths[:,None]
    pair,_=swept_clearances(before,after,np.empty((0,3)),c.agent_radius)
    assert pair > 0
    assert np.all(lengths < c.speed*c.dt)
    # An isolated agent stays at nominal speed: the guard adds no attraction.
    isolated=np.array([[-20.,0],[20.,0]])
    np.testing.assert_allclose(safe_step_lengths(isolated,np.empty((0,3)),c),c.speed*c.dt)


def test_local_braking_preserves_obstacle_clearance():
    c=Config()
    before=np.array([[1.2,0]])
    obstacles=np.array([[0.,0.,1.]])
    lengths=safe_step_lengths(before,obstacles,c)
    after=before+np.array([[-1.,0]])*lengths[:,None]
    _,clearance=swept_clearances(before,after,obstacles,c.agent_radius)
    assert clearance > 0
