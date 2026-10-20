"""Development-only search for recovery from matched spatial and angular damage.

This searches the existing equations, not a redesigned controller. A common
rigid, aligned observation history initializes memory before every intervention.
All candidates and paired outcomes are retained; no confirmatory p-values here.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
from scipy.stats import qmc
from .config import Config
from .controller import bloch_step, geometry, stationary_z
from .compare import equilibrium_tone, endpoints
from .external_controllers import MatchedController, INTERNAL_MODES, PLANT
from .scenarios import make_scenario
from .simulation import simulate
from .storage import prepare_output,save_result,write_csv,write_json
from .storage import digest

TASKS={'moderate':{'heading_sd':.8,'noise':.1},'severe':{'heading_sd':1.6,'noise':.35}}
SPACE={'alignment_gain':(.2,4.,'log'),'cohesion_gain':(.01,.5,'log'),
       'separation_gain':(1.,15.,'log'),'separation_radius':(.6,1.5,'linear'),
       'g_align':(.1,2.,'log'),'gamma':(.1,3.,'log'),'t1':(.2,2.,'log'),
       't2_ratio':(.1,.9,'linear'),'kappa':(.2,6.,'log'),'slow_rate':(.03,1.,'log'),
       'feedback_gain':(1.,6.,'linear'),'caution_gain':(0.,3.,'linear')}


def conditioning_state(c,mode,seed,heading_sd,history_duration=20.,obstacle=False):
    """Exact homogeneous-channel replay of a shared rigid aligned prehistory.

    All active channels see identical headings and fields within an agent;
    every agent has neighbors. Thus one channel + one self value exactly
    reproduce the full matrix update under this prescribed observation history.
    The history is a controlled sensory exposure, not self-generated motion.
    """
    pos,_,groups,_=make_scenario(c,'open',seed)
    _,_,adj=geometry(pos,c.sensing_radius)
    if not np.all(adj.any(axis=1)):raise ValueError('history requires neighbors')
    m=np.zeros(3);s=c.frozen_tone if mode=='frozen' else c.slow_base
    feedback=mode in ('full','frozen','scalar','scalar_matched','instant_matched')
    if mode in INTERNAL_MODES:
        for _ in range(round(history_duration/c.dt)):
            hz=c.g_align+(c.kappa*s if feedback else 0.)
            if mode=='instant_matched':
                z=float(stationary_z(hz,c.gamma,c.t1,c.t2))
            elif mode in ('scalar','scalar_matched'):
                q=np.tanh(hz) if mode=='scalar' else stationary_z(hz,c.gamma,c.t1,c.t2)
                m[2]=q+(m[2]-q)*np.exp(-c.dt/c.t1);z=float(m[2])
            else:
                m=bloch_step(m,np.asarray(hz),c.gamma,c.t1,c.t2,c.dt);z=float(m[2])
            if mode in ('full','scalar','scalar_matched','instant_matched'):
                target=c.slow_base+np.tanh(c.feedback_gain*z)
                s=target+(s-target)*np.exp(-c.slow_rate*c.dt)
    left,right=groups==0,groups==1
    # Equal cohorts are not required; actual damage is verified from the graph.
    pos[left,0]+=-c.split_gap/2-pos[left,0].max()
    pos[right,0]+=c.split_gap/2-pos[right,0].min()
    rng=np.random.default_rng(np.random.SeedSequence([seed,404]))
    theta=np.pi/2+heading_sd*rng.normal(size=c.n)
    theta+=np.where(left,c.split_angle,-c.split_angle)
    obstacles=np.array([[0.,pos[:,1].max()+c.obstacle_first_offset,c.obstacle_radius]]) if obstacle else np.empty((0,3))
    return {'positions':pos,'headings':theta,'groups':groups,'obstacles':obstacles,
            'm':adj[...,None]*m,'s':np.full(c.n,s)}


def sample_candidates(count):
    base=replace(Config.load(ROOT/'data/development/tuned_full.json'),duration=45.,split_gap=4.)
    configs=[base]
    sensitive=replace(base,alignment_gain=1.5,cohesion_gain=.15,separation_gain=6.,
       separation_radius=.9,g_align=.3,kappa=3.,feedback_gain=4.,slow_rate=.3,
       gamma=1.,t1=.8,t2=.25,caution_gain=0.)
    configs += [sensitive,replace(sensitive,alignment_gain=3.,caution_gain=1.5),
                replace(sensitive,gamma=.4,t1=.4,t2=.15,alignment_gain=.7)]
    design=qmc.LatinHypercube(d=len(SPACE),seed=20261020).random(max(0,count-4))
    for row in design:
        values={key:float(np.exp(np.log(lo)+u*np.log(hi/lo)) if scale=='log' else lo+u*(hi-lo))
                for u,(key,(lo,hi,scale)) in zip(row,SPACE.items())}
        values['t2']=values.pop('t2_ratio')*values['t1']
        configs.append(replace(base,**values))
    for c in configs:c.validate()
    return configs[:count]


def run_one(job):
    spec,mode,task,seed,index,out,record=job
    c=replace(Config(**spec),noise=TASKS[task]['noise'])
    c=replace(c,frozen_tone=equilibrium_tone(c))
    state=conditioning_state(c,mode,seed,TASKS[task]['heading_sd'])
    result=simulate(c,'split',seed,mode,record,initial_state=state,
                    controller_factory=MatchedController)
    extra=endpoints(result)
    t,f,p=result.series['time'],result.series['cluster'],result.series['polar']
    joint_deficit=float(np.sum((2-f[:-1]*p[:-1]-f[1:]*p[1:])*np.diff(t)/2)/c.duration)
    progress_loss=max(0.,1-result.summary['progress_fraction'])
    score=.45*joint_deficit+.35*extra['restricted_recovery_time']/c.duration+.2*progress_loss
    score+=2*(not result.summary['safe'])
    result.config.update(plant=PLANT,history_duration=20.,heading_shock_sd=TASKS[task]['heading_sd'])
    save_result(result,Path(out)/f'candidate_{index:03d}'/task/'runs')
    return {'candidate':index,'case':task,**result.summary,**extra,
            'joint_deficit':joint_deficit,'search_loss':float(score),
            'initial_cluster':float(f[0]),'initial_polar':float(p[0]),
            'initial_self':float(state['s'].mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--candidates',type=int,default=48);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();configs=sample_candidates(args.candidates);out=prepare_output(args.out)
    modes=('full','fast','frozen');seeds=[41001,41002]
    protocol={'status':'exploratory development; no significance claims',
      'configs':[c.to_dict() for c in configs],'tasks':TASKS,'seeds':seeds,'controllers':modes,
      'history':'20 units, prescribed rigid aligned lattice; identical physical observations; controller-specific memory',
      'damage':'gap 4 between original cohorts and seeded Gaussian heading shock; no obstacles',
      'plant':PLANT,'space':SPACE,'objective':'.45 mean(1-F*P) + .35 restricted recovery/45 + .2 lost progress +2 unsafe',
      'retention':'all candidates, seeds, controls and failures saved; parameters and tasks may be developed here, not on final test seeds'}
    write_json(out/'protocol.json',protocol);(out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
    jobs=[(c.to_dict(),m,task,seed,i,str(out),i<4 and seed==seeds[0])
          for i,c in enumerate(configs) for task in TASKS for seed in seeds for m in modes]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_one,jobs,chunksize=1):
            rows.append(row)
            if len(rows)%24==0 or len(rows)==len(jobs):
                write_csv(out/'summary.csv',rows);print(f'{len(rows)}/{len(jobs)} development runs',flush=True)
    leaderboard=[]
    for i in range(len(configs)):
        row={'candidate':i}
        for m in modes:
            group=[r for r in rows if r['candidate']==i and r['controller']==m]
            row[m+'_loss']=float(np.mean([r['search_loss'] for r in group]))
            row[m+'_recovery']=float(np.mean([r['restricted_recovery_time'] for r in group]))
            row[m+'_healed']=sum(r['healing_success'] for r in group)
        leaderboard.append(row)
    write_csv(out/'leaderboard.csv',sorted(leaderboard,key=lambda x:x['full_loss']))
    write_json(out/'complete.json',{'runs':len(rows),'protocol_sha256':digest(out/'protocol.json')})

if __name__=='__main__':main()
