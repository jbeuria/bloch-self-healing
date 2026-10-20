"""Fixed-parameter, one-factor-at-a-time extension of the frozen recovery study.

Simulation only. Analysis and PDF plotting live in separate modules. The seven
settings and fresh seeds are saved before any outcome is evaluated.
"""
from pathlib import Path
from dataclasses import replace
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import argparse, hashlib, json, os, time
import numpy as np
from .config import Config
from .external_controllers import MatchedController, PLANT
from .healing_refinement import TASKS
from .healing_search import conditioning_state
from .compare import endpoints
from .simulation import simulate
from .storage import prepare_output, save_result, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
MODES = ('full','olfati_saber','couzin','vasarhelyi','fast','frozen','scalar_matched')
SETTINGS = {
    'reference': {},
    'population_24': {'n':24},
    'population_96': {'n':96},
    'dt_025': {'dt':.025},
    'dt_0125': {'dt':.0125},
    'centering_half': {'center_gain':.0175},
    'centering_zero': {'center_gain':0.},
}
FINE_DT = .0125

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def noise_tape(seed,n,dt,duration):
    """Nested Wiener increments, shared across modes, settings and step sizes.

    Fixed 96-column master tape keeps the first n agent streams unchanged when
    population changes. Population-dependent geometry is deliberately not fixed.
    """
    ratio=round(dt/FINE_DT);steps=round(duration/dt)
    if abs(ratio*FINE_DT-dt)>1e-12 or n>96:raise ValueError('unsupported grid/population')
    fine=np.random.default_rng(np.random.SeedSequence([seed,620202])).normal(size=(steps*ratio,96))[:,:n]
    return fine.reshape(steps,ratio,n).sum(axis=1)/np.sqrt(ratio)

class ReplayController(MatchedController):
    def __init__(self,config,mode,parameters,noise):
        super().__init__(config,mode,parameters=parameters)
        self.noise=noise;self.index=0
    def step(self,pos,theta,obstacles,unused_noise):
        noise=self.noise[self.index];self.index+=1
        return super().step(pos,theta,obstacles,noise)

def run_one(job):
    setting,case,mode,seed,spec,out=job
    c=replace(Config(**spec['config']),**SETTINGS[setting],noise=TASKS[case]['noise'])
    history_mode='full' if mode=='frozen' else mode
    state=conditioning_state(c,history_mode,seed,TASKS[case]['heading_sd'],obstacle=TASKS[case]['obstacle'])
    tape=noise_tape(seed,c.n,c.dt,c.duration)
    factory=partial(ReplayController,parameters=spec['parameters'],noise=tape)
    started=time.perf_counter()
    result=simulate(c,'split',seed,mode,False,initial_state=state,controller_factory=factory)
    elapsed=time.perf_counter()-started
    t,f,p=[result.series[k] for k in ('time','cluster','polar')]
    if f[0]>=c.split_fraction:raise ValueError('initial damage did not split flock')
    extra=endpoints(result)
    joint=float(np.sum((2-f[:-1]*p[:-1]-f[1:]*p[1:])*np.diff(t)/2))
    result.config.update(controller_parameters=spec['parameters'],plant=PLANT,
                         history_duration=20.,heading_shock_sd=TASKS[case]['heading_sd'],
                         noise_scheme='nested master Wiener increments, dt_min=.0125')
    physical_hash=hashlib.sha256(b''.join(np.asarray(state[k]).tobytes() for k in ('positions','headings','groups','obstacles'))).hexdigest()
    internal_hash=hashlib.sha256(state['m'].tobytes()+state['s'].tobytes()).hexdigest()
    save_result(result,Path(out)/setting/case/'runs')
    return {'setting':setting,'case':case,**result.summary,**extra,
            'joint_recovery_deficit':joint,'guard_fraction_mean':float(result.series['guard_fraction'][1:].mean()),
            'actual_acceleration_max':float(result.series['actual_acceleration_max'].max()),
            'initial_cluster':float(f[0]),'initial_polar':float(p[0]),'initial_self':float(state['s'].mean()),
            'n_agents':c.n,'dt':c.dt,'center_gain':c.center_gain,
            'timed_simulation_seconds':elapsed,'physical_initial_sha256':physical_hash,
            'internal_initial_sha256':internal_hash,
            'noise_sha256':hashlib.sha256(tape.tobytes()).hexdigest()}

def evaluate(out,workers=6,seeds=30,first_seed=62001,selection=None):
    selection=Path(selection) if selection is not None else ROOT/'data/tuning/selected.json'
    chosen=json.loads(selection.read_text())
    for mode in ('fast','frozen','scalar_matched'):chosen[mode]=chosen['full']
    chosen={m:chosen[m] for m in MODES}
    seed_list=list(range(first_seed,first_seed+seeds))
    if set(seed_list)&set(list(range(51001,51101))+[41001,41002,43001,43002]+list(range(44001,44007))):
        raise ValueError('fresh seeds must not overlap prior study/development')
    out=prepare_output(out)
    protocol={'status':'prospective supplementary sensitivity protocol, designed after the primary study; no retuning',
        'settings':SETTINGS,'tasks':TASKS,'controllers':MODES,'seeds':seed_list,'selected':chosen,
        'selection_sha256':digest(selection),'horizon':45.,'history_duration':20.,
        'noise':'dt_min=.0125 Gaussian tape with 96 columns, RNG SeedSequence([seed,620202]); aggregate consecutive increments for larger dt; first n columns for population n',
        'design':'one factor at a time; seven unique settings including fresh paired reference; constant lattice spacing, gap, sensing and obstacle radius; population changes footprint, not density',
        'endpoints':['restricted_recovery_time','joint_recovery_deficit'],
        'secondary':['healing_success','progress_fraction','guard_fraction_mean','timed_simulation_seconds','actual_acceleration_max'],
        'statistics':'means and sample SD; paired permutation (50000) and paired-t sensitivity, Holm across 252 supplementary controller contrasts; ordinary paired-bootstrap 95% intervals (20000), not simultaneous; separate 252-test setting-effect family; 126 exact McNemar success contrasts, separate Holm',
        'limits':'fixed radius obstacle is not scaled with N; runtime includes dense diagnostics and safety checks, concurrent workers and Python overhead; dt modifies the complete discrete controller/noise/guard process, not a proof of continuous-time convergence',
        'claim':'No selection of settings or controller parameters based on these outcomes. All runs and failures retained.'}
    write_json(out/'protocol.json',protocol);(out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    jobs=[(setting,case,mode,seed,chosen[mode],str(out)) for setting in SETTINGS for case in TASKS for seed in seed_list for mode in MODES]
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(run_one,jobs,chunksize=1):
            rows.append(row)
            if len(rows)%42==0 or len(rows)==len(jobs):
                write_csv(out/'summary.csv',rows);print(f'generalization: {len(rows)}/{len(jobs)}',flush=True)
    write_json(out/'complete.json',{'runs':len(rows),'protocol_sha256':digest(out/'protocol.json'),'selection_sha256':digest(selection)})

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--seeds',type=int,default=30);ap.add_argument('--first-seed',type=int,default=62001)
    ap.add_argument('--selection',type=Path,help='explicit fixed parameter specification; default original selected configuration');a=ap.parse_args()
    if a.workers<1 or a.seeds<2:ap.error('positive workers and at least two seeds required')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
    evaluate(a.out,a.workers,a.seeds,a.first_seed,a.selection)

if __name__=='__main__':main()
