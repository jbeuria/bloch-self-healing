"""Numerical safeguards for paired sensitivity experiments."""
from dataclasses import replace
import numpy as np
from self_healing.config import Config
from self_healing.generalization import noise_tape, ReplayController
from self_healing.external_controllers import MatchedController
from self_healing.healing_search import conditioning_state

def test_nested_brownian_increments():
    fine=noise_tape(62001,48,.0125,1.)*.0125**.5
    for dt in (.025,.05):
        coarse=noise_tape(62001,48,dt,1.)*dt**.5
        np.testing.assert_allclose(coarse,fine.reshape(-1,round(dt/.0125),48).sum(1),atol=1e-15)
    np.testing.assert_array_equal(noise_tape(62001,24,.025,1.),noise_tape(62001,96,.025,1.)[:,:24])

def test_replay_matches_native_controller_for_same_noise():
    c=replace(Config(),n=24,duration=2.)
    state=conditioning_state(c,'full',62001,.8)
    tape=noise_tape(62001,24,.05,2.)
    a=MatchedController(c,'full',parameters={})
    b=ReplayController(c,'full',parameters={},noise=tape)
    a.restore_internal_state(state['m'],state['s']);b.restore_internal_state(state['m'],state['s'])
    p1=p2=state['positions'];t1=t2=state['headings']
    for k in range(10):
        p1,t1=a.step(p1,t1,state['obstacles'],tape[k])
        p2,t2=b.step(p2,t2,state['obstacles'],np.zeros(c.n))
        np.testing.assert_array_equal(p1,p2);np.testing.assert_array_equal(t1,t2)
    assert b.index==10
