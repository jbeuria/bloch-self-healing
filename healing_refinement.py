"""Task-focused parameter optimization and fresh-seed recovery comparison.

The full-model equations are unchanged. All external models receive the same
final search/validation budget. Search candidates and all outcomes are archived.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
import json
import csv
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
from scipy.stats import qmc
from .config import Config
from .compare import equilibrium_tone,endpoints
from .external_controllers import MatchedController,DEFAULTS,EXTERNAL_MODES,INTERNAL_MODES,PLANT
from .healing_search import conditioning_state,SPACE as FULL_SPACE
from .simulation import simulate
from .storage import prepare_output,save_result,write_csv,write_json
from .storage import digest

TASKS={'moderate':{'heading_sd':.8,'noise':.1,'obstacle':False},
       'severe':{'heading_sd':1.6,'noise':.35,'obstacle':False},
       'obstacle':{'heading_sd':1.2,'noise':.2,'obstacle':True}}
SPACES={
 'full':{**FULL_SPACE,'obstacle_gain':(1.,20.,'log')},
 'olfati_saber':{'spacing':(.8,2.9,'linear'),'potential':(.05,2.,'log'),
                'consensus':(.1,8.,'log'),'repulsion_ratio':(1.,20.,'log'),
                'obstacle_potential':(2.,20.,'log'),'obstacle_damping':(.1,4.,'log'),
                'clearance':(1.,3.,'linear')},
 'couzin':{'repulsion_radius':(.45,1.4,'linear'),'orientation_radius':(1.5,3.1,'linear'),
           'social_gain':(.05,4.,'log'),'obstacle_gain':(1.,20.,'log')},
 'vasarhelyi':{'repulsion_radius':(.45,1.8,'linear'),'repulsion_gain':(.1,8.,'log'),
              'friction_offset':(.5,3.2,'linear'),'friction_gain':(.1,4.,'log'),
              'velocity_slack':(.005,.3,'log'),'braking_acceleration':(.2,4.,'log'),
              'braking_gain':(.2,4.,'log'),'shill_offset':(.05,1.8,'linear'),
              'shill_speed':(1.,4.,'linear'),'shill_acceleration':(.2,4.,'log'),
              'shill_gain':(.2,4.,'log')},
}


def physical_base():
    return replace(Config.load(ROOT/'data/development/tuned_full.json'),duration=45.,split_gap=4.)


def decode(u,space):
    return {key:float(np.exp(np.log(lo)+x*np.log(hi/lo)) if scale=='log' else lo+x*(hi-lo))
            for x,(key,(lo,hi,scale)) in zip(u,space.items())}


def encode(values,space):
    return np.array([(np.log(values[k]/lo)/np.log(hi/lo) if scale=='log' else (values[k]-lo)/(hi-lo))
                     for k,(lo,hi,scale) in space.items()])


def spec_from_values(mode,values):
    base=physical_base()
    if mode=='full':
        values=dict(values);values['t2']=values.pop('t2_ratio')*values['t1']
        c=replace(base,**values);c.validate()
        return {'config':c.to_dict(),'parameters':{}}
    return {'config':base.to_dict(),'parameters':{**DEFAULTS[mode],**values}}


def initial_candidates(count,pilot):
    base=physical_base();previous=json.loads((ROOT/'data/development/external_selected.json').read_text())
    pilot=Path(pilot);old=json.loads((pilot/'protocol.json').read_text())
    leaders=list(csv.DictReader((pilot/'leaderboard.csv').open()))
    specs={}
    for mi,mode in enumerate(EXTERNAL_MODES):
        if mode=='full':
            pool=[{'config':base.to_dict(),'parameters':{}}]
            for row in leaders[:7]:pool.append({'config':old['configs'][int(row['candidate'])],'parameters':{}})
        else:
            pool=[{'config':base.to_dict(),'parameters':DEFAULTS[mode]},
                  {'config':base.to_dict(),'parameters':{**DEFAULTS[mode],**previous[mode]['parameters']}}]
        design=qmc.LatinHypercube(d=len(SPACES[mode]),seed=20261021+mi).random(count-len(pool))
        pool += [spec_from_values(mode,decode(u,SPACES[mode])) for u in design]
        specs[mode]=pool
    return specs


def run_one(job):
    spec,mode,case,seed,index,out,record=job
    task=TASKS[case]
    c=replace(Config(**spec['config']),noise=task['noise'])
    c=replace(c,frozen_tone=equilibrium_tone(c))
    effective='full' if mode=='original_full' else mode
    # Freeze-at-damage control starts from exactly the full model's memory;
    # only subsequent self updates are disabled. No initial-state/gain mismatch.
    history_mode='full' if effective=='frozen' else effective
    state=conditioning_state(c,history_mode,seed,task['heading_sd'],obstacle=task['obstacle'])
    factory=partial(MatchedController,parameters=spec['parameters'])
    result=simulate(c,'split',seed,effective,record,initial_state=state,controller_factory=factory)
    result.summary['controller']=mode
    extra=endpoints(result)
    t,f,p=result.series['time'],result.series['cluster'],result.series['polar']
    joint=float(np.sum((2-f[:-1]*p[:-1]-f[1:]*p[1:])*np.diff(t)/2))
    loss=.4*joint/c.duration+.3*extra['restricted_recovery_time']/c.duration
    loss+=.15*max(0.,1-result.summary['progress_fraction'])+.15*(not result.summary['healing_success'])+2*(not result.summary['safe'])
    result.config.update(controller_parameters=spec['parameters'],plant=PLANT,
                         history_duration=20.,heading_shock_sd=task['heading_sd'])
    save_result(result,Path(out)/case/'runs')
    return {'case':case,'candidate':index,**result.summary,**extra,
            'joint_recovery_deficit':joint,'search_loss':float(loss),
            'guard_fraction_mean':float(result.series['guard_fraction'][1:].mean()),
            'initial_cluster':float(f[0]),'initial_polar':float(p[0]),'initial_self':float(state['s'].mean())}


def execute(jobs,out,workers,name):
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(run_one,jobs,chunksize=1):
            rows.append(row)
            if len(rows)%24==0 or len(rows)==len(jobs):
                write_csv(out/(name+'.csv'),rows);print(f'{name}: {len(rows)}/{len(jobs)}',flush=True)
    return rows


def rank(rows,mode):
    indices=sorted({int(r['candidate']) for r in rows if r['controller']==mode})
    return sorted([{'candidate':i,'controller':mode,
                    'loss':float(np.mean([r['search_loss'] for r in rows if r['candidate']==i and r['controller']==mode]))}
                   for i in indices],key=lambda x:(x['loss'],x['candidate']))


def tune(a):
    out=prepare_output(a.out);specs=initial_candidates(48,a.pilot)
    seeds=[43001,43002];validation_seeds=list(range(44001,44007))
    protocol={'status':'development and parameter selection, no confirmatory tests',
      'controllers':EXTERNAL_MODES,'tasks':TASKS,'initial_candidates':specs,'spaces':SPACES,
      'training_seeds':seeds,'validation_seeds':validation_seeds,'plant':PLANT,
      'budget_per_algorithm':'48 initial + 16 local candidates x 2 seeds x 3 tasks = 384 training runs; best 4 x 6 validation seeds x 3 tasks = 72 validation runs',
      'local_search':'two Gaussian perturbations (SD .15 in normalized parameter coordinates) for each of best 8; clipped to declared ranges',
      'selection':'lowest mean validation loss among four training finalists',
      'loss':'.4 joint recovery deficit/H + .3 restricted first recovery/H + .15 lost progress + .15 not healed +2 unsafe',
      'scope':'existing full equations, common warm observation replay, controlled spatial and angular damage; task family selected during development',
      'prior_development':'full anchors from healing_search_48; external anchors from first external study plus literature settings; final budgets equal, historical effort differs',
      'olfati_extension':'allow published asymmetric sigmoid b/a between 1 and 20, enabling task-specific spacing; same equations and physical limits'}
    write_json(out/'protocol.json',protocol);(out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    def jobs_for(indices,seeds,stage):
        return [(specs[m][i],m,case,seed,i,str(out/stage/m/f'candidate_{i:03d}'),False)
                for i in indices for m in EXTERNAL_MODES for case in TASKS for seed in seeds]
    rows=execute(jobs_for(range(48),seeds,'training'),out,a.workers,'initial_training')
    rng=np.random.default_rng(20261022)
    for mode in EXTERNAL_MODES:
        for item in rank(rows,mode)[:8]:
            spec=specs[mode][item['candidate']]
            values=dict(spec['config'] if mode=='full' else spec['parameters'])
            if mode=='full':values['t2_ratio']=values['t2']/values['t1']
            u=encode(values,SPACES[mode])
            for _ in range(2):
                specs[mode].append(spec_from_values(mode,decode(np.clip(u+rng.normal(0,.15,len(u)),0,1),SPACES[mode])))
    write_json(out/'all_candidates.json',specs)
    rows+=execute(jobs_for(range(48,64),seeds,'training'),out,a.workers,'local_training')
    write_csv(out/'training_summary.csv',rows)
    finalists={m:[x['candidate'] for x in rank(rows,m)[:4]] for m in EXTERNAL_MODES}
    write_json(out/'validation_protocol.json',{'finalists':finalists,'seeds':validation_seeds,
                     'all_candidates_sha256':digest(out/'all_candidates.json')})
    jobs=[(specs[m][i],m,case,seed,i,str(out/'validation'/m/f'candidate_{i:03d}'),False)
          for m in EXTERNAL_MODES for i in finalists[m] for case in TASKS for seed in validation_seeds]
    vr=execute(jobs,out,a.workers,'validation_summary')
    chosen={};leaders=[]
    for m in EXTERNAL_MODES:
        ranked=rank(vr,m);leaders+=ranked;best=ranked[0]
        chosen[m]={**specs[m][best['candidate']],'candidate':best['candidate'],'validation_loss':best['loss']}
    write_json(out/'selected.json',chosen);write_csv(out/'validation_leaderboard.csv',leaders)
    write_json(out/'complete.json',{'training_runs':len(rows),'validation_runs':len(vr),
         'protocol_sha256':digest(out/'protocol.json'),'selected_sha256':digest(out/'selected.json')})


def evaluate(a):
    source=Path(a.tuning);complete=json.loads((source/'complete.json').read_text())
    if complete['selected_sha256']!=digest(source/'selected.json'):raise ValueError('selection changed')
    chosen=json.loads((source/'selected.json').read_text())
    for mode in ('fast','frozen','scalar_matched'):chosen[mode]=chosen['full']
    chosen['original_full']={'config':physical_base().to_dict(),'parameters':{},'candidate':-1}
    seeds=list(range(a.first_seed,a.first_seed+a.seeds))
    train=json.loads((source/'protocol.json').read_text())
    if set(seeds)&set(train['training_seeds']+train['validation_seeds']+[41001,41002]):raise ValueError('seed leakage')
    out=prepare_output(a.out)
    protocol={'status':'frozen held-out confirmation after exploratory task development',
       'controllers':list(chosen),'main_cases':list(TASKS),'tasks':TASKS,'seeds':seeds,
       'selected':chosen,'plant':PLANT,'tuning':str(source.resolve()),'selection_sha256':digest(source/'selected.json'),
       'primary_endpoints':['joint_recovery_deficit','restricted_recovery_time'],
       'primary_family':'42 full-minus-control contrasts: seven controls x three tasks x two endpoints; paired permutations with Holm correction',
       'statistics':'mean +/- sample SD; 20000 paired bootstrap resamples; Bonferroni simultaneous intervals; 50000 sign permutations; paired-t sensitivity',
       'damage':'identical initial positions, split and angular shock for every controller; all cases begin fragmented',
       'history':'20-unit prescribed rigid aligned observation history; controller-specific internal state, identical physical state',
       'frozen_control':'copy full m and s at the damage time, then disable self updates; exact matched initial memory',
       'healing':'F>=.9 and P>=.9 for 2 units; overall success also requires all split events recovered, final recovery, safety, progress>=.5 and obstacle clearance',
       'recording':'spatial snapshots for first 3 seeds per case, fixed in advance; step metrics for all runs',
       'claim_scope':'task-family-specific comparison after development; parameters optimized independently; original full control measures tuning gain, not self attribution'}
    write_json(out/'protocol.json',protocol);(out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    jobs=[(spec,m,case,seed,spec['candidate'],str(out),seed in seeds[:3])
          for case in TASKS for seed in seeds for m,spec in chosen.items()]
    rows=execute(jobs,out,a.workers,'summary')
    write_json(out/'complete.json',{'runs':len(rows),'protocol_sha256':digest(out/'protocol.json')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['tune','evaluate'])
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--pilot',default=str(ROOT/'data/development/full_screen'));p.add_argument('--tuning')
    p.add_argument('--seeds',type=int,default=100);p.add_argument('--first-seed',type=int,default=51001)
    a=p.parse_args()
    if a.stage=='evaluate' and not a.tuning:p.error('--tuning required')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
    (tune if a.stage=='tune' else evaluate)(a)

if __name__=='__main__':main()
