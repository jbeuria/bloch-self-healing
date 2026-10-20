"""Paired statistics for saved, frozen task-focused recovery experiments."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel,t,binomtest
from .comparison_analysis import paired_inference,holm,wilson_interval
from .storage import write_csv,write_json,snapshot_analysis

LABELS={'full':'Optimized full','original_full':'Original full parameters','fast':'No self',
        'frozen':'Self frozen at damage','scalar_matched':'Matched scalar memory',
        'olfati_saber':'Olfati-Saber adaptation','couzin':'Couzin adaptation','vasarhelyi':'Vasarhelyi adaptation'}
ORDER=('full','original_full','fast','frozen','scalar_matched','olfati_saber','couzin','vasarhelyi')


def load(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text());complete=json.loads((root/'complete.json').read_text())
    digest=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
    if digest!=(root/'protocol.sha256').read_text().strip() or digest!=complete['protocol_sha256']:raise ValueError('protocol hash mismatch')
    rows=list(csv.DictReader((root/'summary.csv').open()))
    booleans={'safe','progress_ok','obstacles_cleared','healing_success','had_split','eventually_recovered','first_event_observed'}
    for r in rows:
        for k,v in list(r.items()):
            if k in booleans:
                if v not in ('True','False'):raise ValueError('bad boolean')
                r[k]=v=='True'
            elif k not in ('case','scenario','controller','status'):r[k]=float(v) if v else None
        r['seed']=int(r['seed'])
    lookup={(r['case'],r['controller'],r['seed']):r for r in rows}
    expected={(c,m,s) for c in p['main_cases'] for m in p['controllers'] for s in p['seeds']}
    if len(rows)!=complete['runs'] or len(lookup)!=len(rows) or set(lookup)!=expected:raise ValueError('incomplete/duplicate pairs')
    return rows,lookup,p


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory');a=ap.parse_args();out=Path(a.directory)
    rows,lookup,p=load(out);snapshot_analysis(out);rng=np.random.default_rng(872491)
    family=len(p['main_cases'])*(len(p['controllers'])-1)*len(p['primary_endpoints'])
    aggregates=[];contrasts=[];success=[]
    for case in p['main_cases']:
        for mode in (m for m in ORDER if m in p['controllers']):
            g=[lookup[case,mode,s] for s in p['seeds']];n=len(g);healed=sum(r['healing_success'] for r in g)
            ci=wilson_interval(healed,n)
            row={'case':case,'controller':mode,'n':n,'split_runs':sum(r['had_split'] for r in g),
                 'healed_runs':healed,'healed_ci95_low':ci[0],'healed_ci95_high':ci[1],
                 'safe_runs':sum(r['safe'] for r in g),
                 'mission_runs':sum(r['safe'] and r['progress_ok'] and r['obstacles_cleared'] for r in g)}
            for name in (*p['primary_endpoints'],'fragmentation_burden','search_loss','progress_fraction','guard_fraction_mean','initial_self'):
                values=np.array([r[name] for r in g]);row[name+'_mean']=float(values.mean());row[name+'_sd']=float(values.std(ddof=1))
            aggregates.append(row)
        f=[lookup[case,'full',s] for s in p['seeds']]
        for control in p['controllers']:
            if control=='full':continue
            b=[lookup[case,control,s] for s in p['seeds']]
            for endpoint in p['primary_endpoints']:
                x=np.array([r[endpoint] for r in f]);y=np.array([r[endpoint] for r in b]);d=x-y
                inference=paired_inference(x,y,family,rng)
                if d.std(ddof=1)>1e-14:
                    test=ttest_rel(x,y);pv=float(test.pvalue);half=float(t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
                else:pv=1. if abs(d.mean())<1e-14 else 0.;half=0.
                contrasts.append({'case':case,'control':control,'endpoint':endpoint,**inference,
                    'paired_t_p':pv,'paired_t_ci95_low':float(d.mean()-half),'paired_t_ci95_high':float(d.mean()+half),
                    'difference_sd':float(d.std(ddof=1)),
                    'marked_20pct_supported':bool(inference['improvement_family_low'] is not None and inference['improvement_family_low']>=20)})
            delta=np.array([int(x['healing_success'])-int(y['healing_success']) for x,y in zip(f,b)])
            up,down=int(sum(delta>0)),int(sum(delta<0))
            success.append({'case':case,'control':control,'difference':float(delta.mean()),
                            'rescued':up,'worsened':down,
                            'mcnemar_p':float(binomtest(up,up+down,.5).pvalue) if up+down else 1.})
    for key in ('permutation_p','paired_t_p'):
        for r,pv in zip(contrasts,holm([r[key] for r in contrasts])):r[key+'_holm']=float(pv)
    for r,pv in zip(success,holm([r['mcnemar_p'] for r in success])):r['mcnemar_p_holm']=float(pv)
    write_csv(out/'aggregate.csv',aggregates);write_csv(out/'contrasts.csv',contrasts);write_csv(out/'success_contrasts.csv',success)
    write_json(out/'analysis.json',{'aggregates':aggregates,'contrasts':contrasts,'success':success,'primary_family_size':family})
    tex=[r'\begin{tabular}{llrrr}',r'\hline',
         r'Task & Controller & Recovery time & Joint deficit & Healed \\',r'\hline']
    for r in aggregates:
        tex.append(f"{r['case'].capitalize()} & {LABELS[r['controller']]} & "
                   f"${r['restricted_recovery_time_mean']:.3f} \\pm {r['restricted_recovery_time_sd']:.3f}$ & "
                   f"${r['joint_recovery_deficit_mean']:.3f} \\pm {r['joint_recovery_deficit_sd']:.3f}$ & "
                   f"{r['healed_runs']}/{r['n']} "+r'\\')
    tex += [r'\hline',r'\end{tabular}',
            '% Mean and sample SD across 100 paired seeds; lower is better; all failed recoveries retained at horizon 45.',
            '% External controllers are independently tuned 2D task adaptations. See METHODS.txt for the comparison scope.']
    (out/'results_table.tex').write_text('\n'.join(tex)+'\n')
    lines=['OPTIMIZED RECOVERY: FROZEN HELD-OUT RESULTS','',f"{len(rows)} runs; {len(p['seeds'])} paired seeds per task/controller. Sample SD across seeds.",
       'Joint deficit = integral(1 - largest-component fraction * polarization) dt.',
       'Restricted recovery = first sustained joint F>=.9 and P>=.9 recovery, unrecovered assigned H=45.',
       'Every trial starts physically split with seeded heading disorder after common observation history.',
       'Smaller is better for both primary endpoints. Success also requires safety, progress and obstacle clearance.','']
    for case in p['main_cases']:
        lines.append(case.upper())
        for r in [r for r in aggregates if r['case']==case]:
            lines.append(f"  {LABELS[r['controller']]:29s} recovery {r['restricted_recovery_time_mean']:7.3f} +/- {r['restricted_recovery_time_sd']:6.3f}; joint deficit {r['joint_recovery_deficit_mean']:7.3f} +/- {r['joint_recovery_deficit_sd']:6.3f}; healed {r['healed_runs']}/{r['n']}; mission {r['mission_runs']}/{r['n']}")
        lines.append('')
    lines+=['PAIRED PRIMARY MEAN COMPARISONS',f'Holm correction across all {family} tests; negative difference favors optimized full.',
            'Intervals are approximate simultaneous Bonferroni bootstrap intervals.']
    for r in contrasts:
        lines.append(f"  {r['case']:9s} {r['endpoint']:26s} vs {r['control']:16s}: {r['difference']:+.4f} [{r['family_ci_low']:+.4f}, {r['family_ci_high']:+.4f}]; Holm p={r['permutation_p_holm']:.5g}; paired-t Holm p={r['paired_t_p_holm']:.5g}")
    lines+=['','CLAIM CHECKS']
    for control in (m for m in ORDER if m != 'full' and m in p['controllers']):
        contrasts_control=[r for r in contrasts if r['control']==control]
        wins=sum(r['difference']<0 and r['permutation_p_holm']<.05 for r in contrasts_control)
        losses=sum(r['difference']>0 and r['permutation_p_holm']<.05 for r in contrasts_control)
        marked=sum(r['marked_20pct_supported'] for r in contrasts_control)
        lines.append(f'  vs {control}: {wins} significant benefits, {losses} significant losses, {marked} >=20% benefits supported; {len(contrasts_control)} task-endpoint comparisons.')
    lines+=['','INTERPRETATION',
       'Original-full comparisons measure parameter-optimization gains, not self-feedback causality.',
       'No-self and freeze-at-damage controls use selected full parameters. Frozen self starts with exactly the same full memory and differs only in subsequent self updates.',
       'No-self removes both the self contribution to the Bloch field and the self-dependent avoidance gate. Its contrast alone does not isolate time-varying feedback from static gain changes; the frozen-self contrast addresses this.',
       'The matched scalar control checks whether the Bloch transient adds value beyond a simpler filter with the same stationary response.',
       'The three external algorithms were independently tuned with identical final evaluation budgets: 384 training + 72 validation runs each, as was full.',
       'Historical development effort differs. The task family was deliberately explored before freezing these fresh seeds; this is not evidence of universal superiority.',
       'The full-model equations are unchanged. The Olfati-Saber search includes the published asymmetric attraction/repulsion function, helping it compete over different spacings.',
       'All use the same physical limits, sensing, migration/centering cue, initialization, perturbation streams and safety guard.',
       'Warm-up is prescribed aligned observation replay, not a self-generated motion history. Initial damage is imposed externally and is identical across controllers.',
       'Guard braking can exceed nominal command acceleration. The disturbance is an exogenous initial condition, not a bounded control action.',
       'Report individual endpoints, progress and success alongside the training objective. Do not call a statistical tie proof of equivalence.',
       'These findings apply to selected parameters and the stated task distribution; tuning uncertainty is not included in the seed intervals.']
    (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':main()
