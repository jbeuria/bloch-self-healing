"""Read-only inputs: fixed-parameter sensitivity statistics, audit and compaction."""
from pathlib import Path
from dataclasses import replace
import argparse,csv,json,hashlib,shutil
import numpy as np
from scipy.stats import ttest_rel,binomtest
from .config import Config
from .metrics import detect_events
from .comparison_analysis import paired_inference,holm,wilson_interval
from .storage import write_csv,write_json

BOOL={'safe','progress_ok','obstacles_cleared','healing_success','had_split','eventually_recovered','first_event_observed'}
TEXT={'setting','case','controller','scenario','status','physical_initial_sha256','internal_initial_sha256','noise_sha256'}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text());done=json.loads((root/'complete.json').read_text())
    assert digest(root/'protocol.json')==(root/'protocol.sha256').read_text().strip()==done['protocol_sha256']
    assert done['selection_sha256']==p['selection_sha256']
    rows=list(csv.DictReader((root/'summary.csv').open()))
    for r in rows:
        for k,v in list(r.items()):
            if k in BOOL:
                assert v in ('True','False');r[k]=v=='True'
            elif k not in TEXT:r[k]=float(v) if v else None
        r['seed']=int(r['seed'])
    lookup={(r['setting'],r['case'],r['controller'],r['seed']):r for r in rows}
    expected={(s,c,m,i) for s in p['settings'] for c in p['tasks'] for m in p['controllers'] for i in p['seeds']}
    assert len(rows)==len(lookup)==len(expected)==done['runs'] and set(lookup)==expected
    return rows,lookup,p

def infer(x,y,rng):
    x=np.asarray(x);y=np.asarray(y);d=x-y
    r=paired_inference(x,y,1,rng)
    for k in ('family_ci_low','family_ci_high','improvement_family_low','improvement_family_high'):r.pop(k)
    r['difference_sd']=float(d.std(ddof=1))
    r['paired_t_p']=float(ttest_rel(x,y).pvalue) if d.std(ddof=1)>1e-14 else (1. if abs(d.mean())<1e-14 else 0.)
    return r

def analyze(root,out):
    rows,lookup,p=read(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(620872491);aggregates=[];contrasts=[];effects=[];success=[]
    endpoints=p['endpoints']
    fields=endpoints+['progress_fraction','guard_fraction_mean','timed_simulation_seconds','actual_acceleration_max','fragmentation_burden']
    for setting in p['settings']:
      for case in p['tasks']:
        full=[lookup[setting,case,'full',s] for s in p['seeds']]
        for mode in p['controllers']:
            g=[lookup[setting,case,mode,s] for s in p['seeds']];healed=sum(r['healing_success'] for r in g)
            low,high=wilson_interval(healed,len(g))
            row={'setting':setting,'case':case,'controller':mode,'n':len(g),'healed_runs':healed,'safe_runs':sum(r['safe'] for r in g),'split_runs':sum(r['had_split'] for r in g),'healed_ci95_low':low,'healed_ci95_high':high}
            for key in fields:
                values=np.array([r[key] for r in g]);row[key+'_mean']=float(values.mean());row[key+'_sd']=float(values.std(ddof=1))
            aggregates.append(row)
            if mode!='full':
                for ep in endpoints:
                    contrasts.append({'setting':setting,'case':case,'control':mode,'endpoint':ep,**infer([r[ep] for r in full],[r[ep] for r in g],rng)})
                delta=np.array([int(a['healing_success'])-int(b['healing_success']) for a,b in zip(full,g)])
                up,down=int(sum(delta>0)),int(sum(delta<0))
                success.append({'setting':setting,'case':case,'control':mode,'difference':float(delta.mean()),'rescued':up,'worsened':down,'mcnemar_p':float(binomtest(up,up+down,.5).pvalue) if up+down else 1.})
            if setting!='reference':
                base=[lookup['reference',case,mode,s] for s in p['seeds']]
                for ep in endpoints:
                    effects.append({'setting':setting,'case':case,'controller':mode,'endpoint':ep,**infer([r[ep] for r in g],[r[ep] for r in base],rng)})
    for family in (contrasts,effects):
        for key in ('permutation_p','paired_t_p'):
            for r,v in zip(family,holm([r[key] for r in family])):r[key+'_holm']=float(v)
    for r,v in zip(success,holm([r['mcnemar_p'] for r in success])):r['mcnemar_p_holm']=float(v)
    convergence=[]
    for case in p['tasks']:
      for mode in p['controllers']:
        for ep in endpoints:
            coarse=np.array([lookup['reference',case,mode,s][ep] for s in p['seeds']])
            medium=np.array([lookup['dt_025',case,mode,s][ep] for s in p['seeds']])
            fine=np.array([lookup['dt_0125',case,mode,s][ep] for s in p['seeds']])
            convergence.append({'case':case,'controller':mode,'endpoint':ep,
                'mean_signed_medium_minus_coarse':float((medium-coarse).mean()),
                'mean_signed_fine_minus_medium':float((fine-medium).mean()),
                'mean_absolute_medium_minus_coarse':float(abs(medium-coarse).mean()),
                'mean_absolute_fine_minus_medium':float(abs(fine-medium).mean()),
                'finer_difference_smaller':bool(abs(fine-medium).mean()<abs(medium-coarse).mean())})
    for name,data in [('aggregate',aggregates),('contrasts',contrasts),('setting_effects',effects),('success_contrasts',success),('time_step_differences',convergence)]:write_csv(out/(name+'.csv'),data)
    write_json(out/'analysis.json',{'source_protocol_sha256':digest(Path(root)/'protocol.json'),'analysis_source_sha256':digest(__file__),'primary_extension_family':len(contrasts),'setting_effect_family':len(effects),'success_family':len(success),'bootstrap':20000,'permutations':50000,'seed':620872491,'intervals':'ordinary paired percentile bootstrap 95%, not simultaneous','pvalues':'separate Holm families; paired-t sensitivity separately corrected','status':'supplementary sensitivity; primary study unchanged; fixed parameters; no retuning'})
    lines=['FIXED-PARAMETER GENERALIZATION CHECKS','',f'{len(rows)} trials; {len(p["seeds"])} fresh paired seeds; means +/- sample SD.','Negative full-minus-control difference favors full. Unconfirmed first recoveries are assigned H=45; mission failure alone does not censor first recovery.','']
    for r in aggregates:
        lines.append(f"{r['setting']:16} {r['case']:9} {r['controller']:15} T={r['restricted_recovery_time_mean']:.4f} +/- {r['restricted_recovery_time_sd']:.4f}; J={r['joint_recovery_deficit_mean']:.4f} +/- {r['joint_recovery_deficit_sd']:.4f}; healed={r['healed_runs']}/{r['n']}; U={r['progress_fraction_mean']:.4f} +/- {r['progress_fraction_sd']:.4f}")
    lines+=['','Full-versus-control comparisons: contrasts.csv (252 comparisons).','Setting-versus-reference comparisons: setting_effects.csv (252 comparisons).','Success comparisons: success_contrasts.csv (126 comparisons).','These supplementary families are separate from the primary comparison family.','The extension was designed after the primary results, with new seeds and no parameter selection.']
    (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print(f'Analyzed {len(rows)} runs: {len(aggregates)} groups, {len(contrasts)} controller contrasts, {len(effects)} setting effects.')

def audit(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows,lookup,p=read(root);verified=0;hashes={}
    for key,r in lookup.items():
        setting,case,mode,seed=key
        path=root/setting/case/'runs'/f'split__{mode}__seed{seed}.npz'
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']));c=replace(Config(**p['selected'][mode]['config']),**p['settings'][setting],noise=p['tasks'][case]['noise'])
            for field,value in c.to_dict().items():assert meta['config'][field]==value,(key,field)
            assert meta['config']['controller_parameters']==p['selected'][mode]['parameters']
            c.validate();t,f,polar=[z[k] for k in ('time','cluster','polar')]
            assert len(t)==round(c.duration/c.dt)+1 and abs(t[-1]-45)<1e-12
            assert f[0]<.85 and np.isfinite(f).all() and np.isfinite(polar).all()
            assert f.min()>0 and f.max()<=1 and polar.min()>=0 and polar.max()<=1+1e-12
            assert z['mean_speed'].max()<=c.speed+1e-10 and z['max_bloch_norm'].max()<=1+1e-10
            assert z['command_acceleration_mean'].max()<=PLANT_ACCEL+1e-10
            assert np.max(abs(z['tone']-c.slow_base))<=1+1e-10
            events=detect_events(t,f,polar,c);assert events==meta['events']
            for k,v in meta['summary'].items():
                if k in r:
                    if isinstance(v,(int,float)) and not isinstance(v,bool):np.testing.assert_allclose(v,r[k],rtol=1e-12,atol=1e-12)
                    else:assert v==r[k],(key,k)
            J=float(np.sum((2-f[:-1]*polar[:-1]-f[1:]*polar[1:])*np.diff(t)/2))
            T=events[0]['duration'] if events[0]['observed'] else c.duration
            np.testing.assert_allclose([J,T],[r['joint_recovery_deficit'],r['restricted_recovery_time']],rtol=1e-12,atol=1e-12)
            final=t>=45-c.hold_time-1e-9
            heal=all(e['observed'] for e in events) and np.all(f[final]>=.9) and np.all(polar[final]>=.9) and r['safe'] and r['progress_ok'] and r['obstacles_cleared']
            assert bool(heal)==r['healing_success']
            assert r['safe']==(r['min_pair_clearance']>=-1e-10 and (r['min_obstacle_clearance'] is None or r['min_obstacle_clearance']>=-1e-10))
            np.testing.assert_allclose((z['centroid_y'][-1]-z['centroid_y'][0])/45,r['progress_fraction'],atol=1e-12)
        full=lookup[setting,case,'full',seed]
        assert r['physical_initial_sha256']==full['physical_initial_sha256'] and r['noise_sha256']==full['noise_sha256']
        if mode=='frozen':assert r['internal_initial_sha256']==full['internal_initial_sha256']
        if setting not in ('population_24','population_96'):
            assert r['physical_initial_sha256']==lookup['reference',case,mode,seed]['physical_initial_sha256']
        hashes[str(path.relative_to(root))]=digest(path);verified+=1
    write_json(out/'raw_manifest.json',hashes)
    write_json(out/'audit.json',{'status':'passed','runs_verified':verified,'protocol_sha256':digest(root/'protocol.json'),'initial_pairing':'all physical and noise hashes matched across controllers; exact full/frozen channel and self states matched','checks':'events, T/J, outcomes, summary consistency, progress, internal bounds, mean speed and command acceleration, stored minimum clearances','limitation':'full spatial paths were not recorded; stored minimum swept clearances come from the simulation safety checker; no independent geometric replay'})
    print(f'Audited all {verified} full raw step records.')

PLANT_ACCEL=2.

def pack(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows,lookup,p=read(root)
    for name in ('protocol.json','protocol.sha256','complete.json','summary.csv','aggregate.csv','contrasts.csv','setting_effects.csv','success_contrasts.csv','time_step_differences.csv','analysis.json','RESULTS.txt','audit.json','raw_manifest.json'):
        shutil.copy2(root/name,out/name)
    hashes={}
    for setting in p['settings']:
        data={};shape=(len(p['tasks']),len(p['controllers']),len(p['seeds']))
        for ci,case in enumerate(p['tasks']):
          for mi,mode in enumerate(p['controllers']):
            for si,seed in enumerate(p['seeds']):
                with np.load(root/setting/case/'runs'/f'split__{mode}__seed{seed}.npz') as z:
                    if not data:
                        data={'time':z['time'],'cluster':np.empty(shape+(len(z['time']),)),'polar':np.empty(shape+(len(z['time']),))}
                    for k in ('cluster','polar'):data[k][ci,mi,si]=z[k]
        path=out/(setting+'_traces.npz');np.savez_compressed(path,**data);hashes[path.name]=digest(path)
    write_json(out/'compact_manifest.json',{'source_protocol_sha256':digest(root/'protocol.json'),'files':hashes,'axis_order':'task, controller, seed, time, in protocol order','retained':'all trial summary fields; all F/P step series; original full raw audit and manifest','omitted':'secondary step diagnostics and redundant metadata, regenerable with generalization command'})
    print('Packed lossless primary traces and all outcomes.')

def check_compact(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows,lookup,p=read(root)
    manifest=json.loads((root/'compact_manifest.json').read_text());count=0
    assert digest(root/'protocol.json')==manifest['source_protocol_sha256']
    selection=root.parent/'tuning/selected.json'
    if not selection.exists():selection=root.parent/'selected.json'
    if selection.exists():assert digest(selection)==p['selection_sha256']
    for setting in p['settings']:
        name=setting+'_traces.npz';assert digest(root/name)==manifest['files'][name]
        with np.load(root/name) as z:
          # Decompress each array once; indexing NpzFile repeatedly re-inflates it.
          times,clusters,polars=z['time'],z['cluster'],z['polar']
          for ci,case in enumerate(p['tasks']):
           for mi,mode in enumerate(p['controllers']):
            for si,seed in enumerate(p['seeds']):
                r=lookup[setting,case,mode,seed];t=times;f=clusters[ci,mi,si];pol=polars[ci,mi,si]
                c=replace(Config(**p['selected'][mode]['config']),**p['settings'][setting])
                events=detect_events(t,f,pol,c);assert f[0]<.85
                full=lookup[setting,case,'full',seed]
                assert r['physical_initial_sha256']==full['physical_initial_sha256']
                assert r['noise_sha256']==full['noise_sha256']
                if mode=='frozen':assert r['internal_initial_sha256']==full['internal_initial_sha256']
                np.testing.assert_allclose([f[0],pol[0]],[full['initial_cluster'],full['initial_polar']],atol=1e-12,rtol=1e-12)
                T=events[0]['duration'] if events[0]['observed'] else 45.
                J=np.sum((2-f[:-1]*pol[:-1]-f[1:]*pol[1:])*np.diff(t)/2)
                np.testing.assert_allclose([T,J],[r['restricted_recovery_time'],r['joint_recovery_deficit']],atol=1e-12,rtol=1e-12)
                final=t>=45-c.hold_time-1e-9
                healed=all(e['observed'] for e in events) and np.all(f[final]>=.9) and np.all(pol[final]>=.9) and r['safe'] and r['progress_ok'] and r['obstacles_cleared']
                assert bool(healed)==r['healing_success'];count+=1
    write_json(out/'generalization_compact_check.json',{'status':'passed','trials_verified':count,'scope':'F/P-based recovery events and primary endpoints; overall healing uses archived safety/progress/clearance classifications'})
    print(f'Compact check: {count} primary traces passed.')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['analyze','audit','pack','check']);ap.add_argument('directory',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
    {'analyze':analyze,'audit':audit,'pack':pack,'check':check_compact}[a.stage](a.directory,a.out or a.directory)

if __name__=='__main__':main()
