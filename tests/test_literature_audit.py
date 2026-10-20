"""Independent source-equation and upstream-C checks for the three adaptations.

Olfati-Saber: Eqs. 8--11, 15, 55--59 and Lemma 4 (2006).
Couzin et al.: Eqs. 1--3 and behavioral rules, p. 3 (2002).
Vasarhelyi et al.: Eqs. 2--11 (2018), using the authors' code for damping signs.
The scalar oracles deliberately avoid all production geometry/interaction helpers.
"""
from dataclasses import replace
from pathlib import Path
import ctypes, json, math, shutil, subprocess
import numpy as np
import pytest
from self_healing.config import Config
from self_healing.external_controllers import (
    DEFAULTS, MatchedController, couzin_social, olfati_acceleration,
    vasarhelyi_velocity, braking_curve, relax_difference,
)

from self_healing.paths import ROOT
SELECTED=json.loads((ROOT/'configurations/working.json').read_text())

def sigma(r):return (math.sqrt(1+.1*r*r)-1)/.1

def taper(x,h):
    if x<h:return 1.
    if x>=1:return 0.
    return (1+math.cos(math.pi*(x-h)/(1-h)))/2

def olfati_from_paper(x,v,obs,c,p):
    out=np.zeros_like(x)
    for i in range(len(x)):
        for j in range(len(x)):
            if j==i:continue
            displacement=x[j]-x[i];r=float(np.linalg.norm(displacement))
            if r>=c.sensing_radius:continue
            w=taper(sigma(r)/sigma(c.sensing_radius),.2)
            a=5.;b=5*p['repulsion_ratio'];offset=abs(a-b)/math.sqrt(4*a*b)
            z=sigma(r)-sigma(p['spacing'])+offset
            phi=((a+b)*z/math.sqrt(1+z*z)+a-b)/2
            out[i]+=p['potential']*w*phi*displacement/math.sqrt(1+.1*r*r)
            out[i]+=p['consensus']*w*(v[j]-v[i])
        for ox,oy,radius in obs:
            rel=x[i]-[ox,oy];r=float(np.linalg.norm(rel));normal=rel/r
            surface=radius+c.agent_radius;gap=r-surface
            assert gap>=0
            if gap>=c.obstacle_margin:continue
            qbeta=np.array([ox,oy])+surface*normal
            w=taper(sigma(gap)/sigma(p['clearance']),.9)
            z=sigma(gap)-sigma(p['clearance']);phi=w*(z/math.sqrt(1+z*z)-1)
            out[i]+=p['obstacle_potential']*phi*(qbeta-x[i])/math.sqrt(1+.1*gap*gap)
            projection=np.eye(2)-np.outer(normal,normal)
            vbeta=surface/r*projection@v[i]
            out[i]+=p['obstacle_damping']*w*(vbeta-v[i])
        out[i]+=[-c.center_gain*x[i,0],0]
        out[i]+=c.migration_gain*(np.array([0.,c.speed])-v[i])
    return out

def couzin_from_paper(x,h,R,rr,ro):
    output=[]
    for i in range(len(x)):
        rep=[];orient=[];attract=[]
        for j in range(len(x)):
            if i==j:continue
            d=x[j]-x[i];r=np.linalg.norm(d)
            if r>=R:continue
            if r<rr:rep.append(-d/r)
            elif r<ro:orient.append(h[j])
            else:attract.append(d/r)
        if rep:d=np.sum(rep,axis=0)
        else:
            d=np.zeros(2)
            for a in orient+attract:d+=a
            if orient and attract:d/=2
        if np.linalg.norm(d)<1e-12:d=h[i]
        output.append(d/np.linalg.norm(d))
    return np.array(output)

def fixtures(seed):
    rng=np.random.default_rng(seed)
    x=rng.uniform(-2.3,2.3,(9,2));theta=rng.uniform(-math.pi,math.pi,9)
    h=np.column_stack([np.cos(theta),np.sin(theta)])
    v=rng.uniform(.05,1,(9,1))*h
    obs=np.array([[4.,0.,.6],[-4.,1.,.4]])
    return x,v,h,obs

@pytest.mark.parametrize('mode',['olfati_saber','couzin'])
@pytest.mark.parametrize('selected',[False,True])
def test_independent_paper_equations_on_random_geometries(mode,selected):
    c=Config(**SELECTED[mode]['config']);p=SELECTED[mode]['parameters'] if selected else DEFAULTS[mode]
    for seed in range(100):
        x,v,h,obs=fixtures(73000+seed)
        if mode=='olfati_saber':
            actual=olfati_acceleration(x,v,obs,c,p);expected=olfati_from_paper(x,v,obs,c,p)
        else:
            args=(x,h,c.sensing_radius,p['repulsion_radius'],p['orientation_radius'])
            actual=couzin_social(*args);expected=couzin_from_paper(*args)
        np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)

def test_couzin_uses_sums_not_separately_normalized_zone_vectors():
    # Two upward orientation neighbors plus one rightward attraction neighbor.
    x=np.array([[0.,0.],[0.,1.2],[0.,-1.2],[2.5,0.]])
    h=np.tile([0.,1.],(4,1))
    actual=couzin_social(x,h,3.2,.8,2.)
    np.testing.assert_allclose(actual[0],np.array([1.,2.])/math.sqrt(5))

def test_couzin_boundaries_and_zero_resultant():
    h=np.tile([0.,1.],(3,1))
    np.testing.assert_allclose(couzin_social(np.array([[0.,0.],[.8,0.]]),h[:2],3.2,.8,2.)[0],[0,1])
    np.testing.assert_allclose(couzin_social(np.array([[0.,0.],[2.,0.]]),h[:2],3.2,.8,2.)[0],[1,0])
    np.testing.assert_allclose(couzin_social(np.array([[0.,0.],[3.2,0.]]),h[:2],3.2,.8,2.)[0],[0,1])
    np.testing.assert_allclose(couzin_social(np.array([[0.,0.],[.5,0.],[-.5,0.]]),h,3.2,.8,2.)[0],[0,1])

def test_olfati_beta_projection_is_derivative_of_nearest_surface_point():
    x=np.array([2.1,-.7]);v=np.array([.4,.9]);radius=.8;dt=1e-5
    project=lambda q: radius*q/np.linalg.norm(q)
    numerical=(project(x+dt*v)-project(x-dt*v))/(2*dt)
    n=x/np.linalg.norm(x);expected=radius/np.linalg.norm(x)*(np.eye(2)-np.outer(n,n))@v
    np.testing.assert_allclose(numerical,expected,rtol=1e-9,atol=1e-10)

def test_olfati_obstacle_repulsion_and_projected_damping():
    c=replace(Config(),center_gain=0,migration_gain=0);p=dict(DEFAULTS['olfati_saber'])
    x=np.array([[1.4,0.]]);obs=np.array([[0.,0.,1.]])
    a=olfati_acceleration(x,np.zeros((1,2)),obs,c,p)
    assert a[0,0]>0 and a[0,1]==0
    v=np.array([[.7,.8]])
    force=olfati_acceleration(x,v,obs,c,p)-a
    assert np.sum(force*v)<0
    assert force[0,1]<0  # sphere curvature gives tangential damping too

@pytest.fixture(scope='module')
def upstream_c(tmp_path_factory):
    compiler=shutil.which('cc')
    if not compiler:pytest.skip('C compiler needed only for the optional upstream source oracle')
    folder=Path(__file__).parent/'fixtures/robotsim';out=tmp_path_factory.mktemp('robotsim')/'oracle.so'
    subprocess.run([compiler,'-shared','-fPIC','-O2',str(folder/'harness.c'),'-lm','-o',str(out)],check=True)
    lib=ctypes.CDLL(str(out));ptr=ctypes.POINTER(ctypes.c_double);dbl=ctypes.c_double
    lib.audit_friction.argtypes=[ctypes.c_int,ptr,ptr,dbl,dbl,dbl,dbl,dbl,ptr]
    lib.audit_shill.argtypes=[ctypes.c_int,ptr,ptr,ptr,dbl,dbl,dbl,dbl,ptr]
    lib.VelDecayLinSqrt.argtypes=[dbl]*5;lib.VelDecayLinSqrt.restype=dbl
    return lib

def as3(x):return np.ascontiguousarray(np.column_stack([x,np.zeros(len(x))]),dtype=np.float64)
def pointer(a):return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

@pytest.mark.parametrize('selected',[False,True])
def test_vasarhelyi_matches_authors_c_interaction_bodies(upstream_c,selected):
    # Large sensing range and one small disk isolate upstream local interactions.
    c=replace(Config(),sensing_radius=100.,obstacle_margin=100.,center_gain=0,migration_gain=0)
    p=SELECTED['vasarhelyi']['parameters'] if selected else DEFAULTS['vasarhelyi']
    for seed in range(100):
        x,v,h,_=fixtures(74000+seed);obs=np.array([[4.,0.,.6]])
        xx=as3(x);vv=as3(v);fr=np.zeros_like(xx);sh=np.zeros_like(xx)
        upstream_c.audit_friction(len(x),pointer(xx),pointer(vv),p['friction_gain'],p['velocity_slack'],p['braking_acceleration'],p['braking_gain'],p['friction_offset'],pointer(fr))
        disk=np.array([4.,0.,.6+c.agent_radius])
        upstream_c.audit_shill(len(x),pointer(xx),pointer(vv),pointer(disk),p['shill_speed'],p['shill_offset'],p['shill_acceleration'],p['shill_gain'],pointer(sh))
        expected=c.speed*h+fr[:,:2]+sh[:,:2]
        for i in range(len(x)):
            for j in range(len(x)):
                if i==j:continue
                d=x[i]-x[j];r=np.linalg.norm(d)
                if r<p['repulsion_radius']:expected[i]+=p['repulsion_gain']*(p['repulsion_radius']-r)*d/r
        np.testing.assert_allclose(vasarhelyi_velocity(x,v,h,obs,c,p),expected,rtol=3e-12,atol=3e-12)

def test_braking_curve_against_upstream_and_damping_energy(upstream_c):
    rng=np.random.default_rng(75000)
    for _ in range(300):
        a,p=rng.uniform(.1,5,2);r=rng.uniform(-2,10);limit=rng.uniform(0,5)
        expected=upstream_c.VelDecayLinSqrt(r,p,a,limit,0.)
        np.testing.assert_allclose(min(braking_curve(r,a,p),limit),expected,rtol=1e-12,atol=1e-12)
        vi,vj=rng.normal(size=(2,2));f=relax_difference(vj-vi,.2)
        # Paired correction removes kinetic energy from relative motion.
        assert np.dot(vi,f)+np.dot(vj,-f)<=1e-12

def test_vasarhelyi_sensor_cutoffs_zero_difference_and_shill_direction():
    c=replace(Config(),center_gain=0,migration_gain=0);p=DEFAULTS['vasarhelyi']
    x=np.array([[0.,0.],[c.sensing_radius,0.]]);h=np.array([[1.,0.],[-1.,0.]])
    np.testing.assert_allclose(vasarhelyi_velocity(x,h,h,np.empty((0,3)),c,p),h)
    x=np.array([[1.4,0.]]);v=np.array([[-1.,0.]]);obs=np.array([[0.,0.,1.]])
    out=vasarhelyi_velocity(x,v,v,obs,c,p)
    assert out[0,0]>v[0,0] and out[0,1]==0
    x=np.array([[1+c.agent_radius+c.obstacle_margin,0.]])
    np.testing.assert_allclose(vasarhelyi_velocity(x,v,v,obs,c,p),v)
    np.testing.assert_array_equal(relax_difference(np.zeros((3,2)),np.ones(3)),np.zeros((3,2)))

@pytest.mark.parametrize('mode',['olfati_saber','couzin','vasarhelyi'])
def test_translation_rotation_covariance_without_navigation(mode):
    x,v,h,obs=fixtures(76000);c=replace(Config(),center_gain=0,migration_gain=0);p=DEFAULTS[mode]
    a=.713;rot=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    shift=np.array([4.7,-2.3]);obs2=obs.copy();obs2[:,:2]=obs[:,:2]@rot.T+shift
    def run(x,v,h,obs):
        if mode=='olfati_saber':return olfati_acceleration(x,v,obs,c,p)
        if mode=='couzin':return couzin_social(x,h,c.sensing_radius,p['repulsion_radius'],p['orientation_radius'])
        return vasarhelyi_velocity(x,v,h,obs,c,p)
    np.testing.assert_allclose(run(x@rot.T+shift,v@rot.T,h@rot.T,obs2),run(x,v,h,obs)@rot.T,rtol=2e-12,atol=2e-12)
