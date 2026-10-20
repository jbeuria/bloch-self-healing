from dataclasses import replace
from functools import partial
import numpy as np
import pytest
from self_healing.config import Config
from self_healing.external_controllers import (MatchedController, DEFAULTS, EXTERNAL_MODES,
    braking_curve, relax_difference, couzin_social, olfati_acceleration)
from self_healing.simulation import simulate


def test_braking_curve_continuity_and_braking_direction():
    a,p = 2.,3.
    cross = a/p**2
    value = braking_curve(np.array([-1,0,cross-1e-8,cross,cross+1e-8]),a,p)
    np.testing.assert_allclose(value[:2],0)
    np.testing.assert_allclose(value[2:],a/p,atol=1e-7)
    dv=np.array([[2.,0.]])
    correction=relax_difference(dv,np.array([.3]))
    assert np.dot(correction[0],dv[0])>0
    np.testing.assert_allclose(correction,[[1.7,0]])


def test_couzin_repulsion_has_priority_over_orientation_and_attraction():
    pos=np.array([[0.,0.],[.5,0],[0,1.5],[2.5,0]])
    headings=np.tile([1.,0.],(4,1))
    out=couzin_social(pos,headings,3.2,.8,2.)
    np.testing.assert_allclose(out[0],[-1.,0.])


def test_olfati_potential_changes_sign_at_spacing_and_consensus_damps():
    c=replace(Config(),center_gain=0,migration_gain=0)
    p=DEFAULTS['olfati_saber']
    for distance,sign in ((.8,-1),(3.,1)):
        pos=np.array([[0.,0.],[distance,0.]])
        a=olfati_acceleration(pos,np.zeros((2,2)),np.empty((0,3)),c,p)
        assert a[0,0]*sign>0
        np.testing.assert_allclose(a.sum(axis=0),0,atol=1e-12)
    pos=np.array([[0.,0.],[p['spacing'],0.]])
    v=np.array([[1.,0.],[-1.,0.]])
    a=olfati_acceleration(pos,v,np.empty((0,3)),c,p)
    assert np.sum(a*v)<0


@pytest.mark.parametrize('mode',EXTERNAL_MODES)
def test_shared_plant_speed_turn_acceleration_and_reproducibility(mode):
    c=replace(Config(),n=8,duration=2,record_every=1,initial_spacing=2.)
    factory=partial(MatchedController,parameters=DEFAULTS[mode])
    a=simulate(c,'open',19001,mode,controller_factory=factory)
    b=simulate(c,'open',19001,mode,controller_factory=factory)
    np.testing.assert_array_equal(a.positions,b.positions)
    v=np.diff(a.positions,axis=0)/c.dt
    assert np.linalg.norm(v,axis=-1).max()<=c.speed+1e-12
    assert np.linalg.norm(np.diff(v,axis=0),axis=-1).max()/c.dt<=2.+1e-9
    turn=(np.diff(a.headings,axis=0)+np.pi)%(2*np.pi)-np.pi
    assert abs(turn).max()<=c.max_turn_rate*c.dt+1e-12
    assert a.summary['safe']


def test_directional_guard_allows_separation_at_contact_and_protects_swept_paths():
    from self_healing.external_controllers import directional_guard
    from self_healing.metrics import swept_clearances
    c=replace(Config(),n=8)
    pos=np.array([[0.,0.],[2*c.agent_radius,0.]])
    v=np.array([[-1.,0.],[1.,0.]])
    actual,guarded=directional_guard(pos,v,np.empty((0,3)),c)
    np.testing.assert_array_equal(actual,v)
    assert not guarded.any()
    rng=np.random.default_rng(181)
    # Near-contact noncollinear velocities, including approaching obstacles.
    pos=np.array([[0.,0.],[.4,0.],[.2,.4],[1.,1.]])
    obs=np.array([[2.,0.,.6]])
    for _ in range(100):
        v=rng.normal(size=pos.shape)*5
        actual,_=directional_guard(pos,v,obs,c)
        pc,oc=swept_clearances(pos,pos+c.dt*actual,obs,c.agent_radius)
        assert pc>=-1e-12 and oc>=-1e-12


def test_homogeneous_history_matches_full_matrix_replay_and_restores_inner_state():
    from self_healing.healing_search import conditioning_state
    from self_healing.controller import Controller
    from self_healing.scenarios import make_scenario
    c=replace(Config(),n=8,duration=2.)
    pos,_,_,_=make_scenario(c,'open',42000)
    theta=np.full(c.n,np.pi/2)
    ctrl=Controller(c,'full')
    for _ in range(40):ctrl.heading_command(pos,theta,np.empty((0,3)),np.zeros(c.n))
    state=conditioning_state(c,'full',42000,.8,history_duration=2.)
    np.testing.assert_allclose(state['s'],ctrl.s,atol=1e-13)
    np.testing.assert_allclose(state['m'],ctrl.m,atol=1e-13)
    matched=MatchedController(c,'full')
    matched.restore_internal_state(state['m'],state['s'])
    np.testing.assert_array_equal(matched.inner.s,state['s'])


@pytest.mark.parametrize('ratio',[1.,5.,20.])
def test_asymmetric_olfati_sigmoid_has_zero_force_at_desired_spacing(ratio):
    c=replace(Config(),center_gain=0,migration_gain=0)
    p={**DEFAULTS['olfati_saber'],'spacing':1.1,'repulsion_ratio':ratio}
    pos=np.array([[0.,0.],[p['spacing'],0.]])
    a=olfati_acceleration(pos,np.zeros((2,2)),np.empty((0,3)),c,p)
    np.testing.assert_allclose(a,0,atol=1e-12)
