"""Build manuscript tables and PDF figures from completed fixed-parameter results only."""
from pathlib import Path
import csv,json,hashlib,runpy,os
from decimal import Decimal,ROUND_HALF_UP
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(os.environ.get('HEALING_ASSET_OUT', str(ROOT/'manuscript')))
HERE.mkdir(parents=True,exist_ok=True)
import sys
sys.path.insert(0,str(ROOT))
from self_healing.bundle import load_run, digest
DATA=Path(os.environ.get('HEALING_DATA',str(ROOT/'results/primary')))
import self_healing.bundle as bundle
bundle.DATA=DATA
FIG=HERE/'figures';TAB=HERE/'tables'
FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
assert json.loads((DATA/'verification.json').read_text())['status']=='passed'
P=json.loads((DATA/'protocol.json').read_text())
FAMILY=len(P['main_cases'])*(len(P['controllers'])-1)*len(P['primary_endpoints'])
assert hashlib.sha256((DATA/'protocol.json').read_bytes()).hexdigest()==(DATA/'protocol.sha256').read_text().strip()
def rows(name):return list(csv.DictReader((DATA/name).open()))
A={(r['case'],r['controller']):r for r in rows('aggregate.csv')}
C={(r['case'],r['control'],r['endpoint']):r for r in rows('contrasts.csv')}
CASES=['moderate','severe','obstacle']
LABEL={'full':'Full','original_full':'Pre-optimization full','fast':'No self feedback','frozen':'Frozen self','scalar_matched':'Matched scalar','olfati_saber':'Olfati-Saber','couzin':'Couzin','vasarhelyi':r'V\'{a}s\'{a}rhelyi'}
PLABEL={'full':'Full','original_full':'Pre-optimization','fast':'No feedback','frozen':'Frozen self','scalar_matched':'Matched scalar','olfati_saber':'Olfati-Saber','couzin':'Couzin','vasarhelyi':'Vásárhelyi'}
COL={'full':'#086f83','olfati_saber':'#d35d12','couzin':'#8053a0','vasarhelyi':'#618f28','fast':'#30363b','frozen':'#ab7b32','scalar_matched':'#ad537e'}
EXT=['full','olfati_saber','couzin','vasarhelyi']
INT=[m for m in ['full','original_full','fast','frozen','scalar_matched'] if m in P['controllers']]
def num(value,n=3):
    return str(Decimal(f'{float(value):.10f}').quantize(Decimal(10)**-n,rounding=ROUND_HALF_UP))
def ms(r,key,n=3):return '$'+num(r[key+'_mean'],n)+r'\,\pm\,'+num(r[key+'_sd'],n)+'$'
def esc(s):return s.replace('_',r'\_')
def pv(v):
    x=float(v)
    if x>=.001:return f'{x:.3g}'
    e=int(np.floor(np.log10(x))) if x else -99
    return '$'+f'{x/10**e:.2f}'+r'\times10^{'+str(e)+'}$' if x else '$<10^{-99}$'

def table(name,modes,progress=False):
    lines=[r'\begin{tabular}{llrr'+('rr' if progress else '')+'}',r'\toprule',
           'Task & Controller & $T_{\\rm rec}$ & $J$'+(' & Healed & $U$' if progress else '')+r' \\',r'\midrule']
    for ci,case in enumerate(CASES):
        if ci:lines.append(r'\midrule')
        for j,m in enumerate(modes):
            r=A[case,m]
            line=(case.capitalize() if j==0 else '')+' & '+LABEL[m]+' & '+ms(r,'restricted_recovery_time')+' & '+ms(r,'joint_recovery_deficit')
            if progress:line+=' & '+r['healed_runs']+'/100 & '+ms(r,'progress_fraction')
            lines.append(line+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (TAB/name).write_text('\n'.join(lines)+'\n')
table('external.tex',EXT,True);table('internal.tex',INT)

# All 42 primary comparisons, in the paper's external-first order.
lines=[r'\begin{longtable}{lllrrrr}',r'\caption{All primary paired comparisons. $\Delta$ is full minus control; negative favors full. Intervals are approximate simultaneous 95\% bootstrap intervals across 42 contrasts. $p_{\rm P}$ and $p_t$ are Holm-adjusted permutation and paired-$t$ probabilities, respectively.}\\',r'\toprule',r'Task & Control & Metric & $\Delta$ & Interval & $p_{\rm P}$ & $p_t$ \\',r'\midrule\endfirsthead',r'\toprule Task & Control & Metric & $\Delta$ & Interval & $p_{\rm P}$ & $p_t$ \\\midrule\endhead',r'\bottomrule\endfoot']
short={'original_full':'Pre-opt.','fast':'No feedback','scalar_matched':'Scalar','frozen':'Frozen',**{m:LABEL[m] for m in EXT}}
for case in CASES:
 for m in EXT[1:]+INT[1:]:
  for ep,sym in [('restricted_recovery_time',r'$T_{\rm rec}$'),('joint_recovery_deficit','$J$')]:
   r=C[case,m,ep]
   lines.append(f"{case.capitalize()} & {short[m]} & {sym} & {num(r['difference'],4)} & [{num(r['family_ci_low'],4)}, {num(r['family_ci_high'],4)}] & {pv(r['permutation_p_holm'])} & {pv(r['paired_t_p_holm'])}"+r' \\')
lines.append(r'\end{longtable}');(TAB/'all_contrasts.tex').write_text(('\n'.join(lines)+'\n').replace('42 contrasts',f'{FAMILY} contrasts'))

selected=json.loads(Path(os.environ.get('HEALING_SELECTION',str(ROOT/'configurations/working.json'))).read_text());tuning=json.loads((ROOT/'configurations/development_spaces.json').read_text())
values=selected['full']['config']
fields=[('k_a','alignment_gain'),('k_c','cohesion_gain'),('k_r','separation_gain'),('r_s','separation_radius'),('k_o','obstacle_gain'),('g_a','g_align'),(r'\Gamma','gamma'),('T_1','t1'),('T_2','t2'),(r'\kappa','kappa'),(r'\gamma_s','slow_rate'),(r'\lambda_{\rm fb}','feedback_gain'),('c_q','caution_gain'),(r's_{\rm base}','slow_base')]
lines=[r'\begin{tabular}{lrlr}',r'\toprule Symbol & Value & Symbol & Value \\\midrule']
for (sym,key),(sym2,key2) in zip(fields[:7],fields[7:]):
 lines.append(f'${sym}$ & {values[key]:.6g} & ${sym2}$ & {values[key2]:.6g}'+r' \\')
lines += [r'\bottomrule',r'\end{tabular}']
(TAB/'selected_full.tex').write_text('\n'.join(lines)+'\n')
lines=[]
for mode in EXT:
 lines += [r'\subsection{'+LABEL[mode]+' search and selected values}',r'\begin{longtable}{lrrl}',r'\caption{Search intervals and selected values for '+LABEL[mode]+r'. Log indicates uniform sampling in logarithmic coordinates.}\\',r'\toprule Parameter & Search interval & Selected & Scale \\\midrule\endfirsthead',r'\toprule Parameter & Search interval & Selected & Scale \\\midrule\endhead',r'\bottomrule\endfoot']
 for key,(lo,hi,scale) in tuning['spaces'][mode].items():
  values=selected[mode]['config'] if mode=='full' else selected[mode]['parameters']
  value=values['t2']/values['t1'] if key=='t2_ratio' else values[key]
  displayed = '$5/9$' if key=='t2_ratio' and values['t1']==.45 and values['t2']==.25 else f'{value:.7g}'
  lines.append(r'\texttt{'+esc(key)+'} & '+f'[{lo:g}, {hi:g}] & {displayed} & {scale}'+r' \\')
 lines.append(r'\end{longtable}')
(TAB/'parameters.tex').write_text(('\n'.join(lines)+'\n').replace('search and selected values','development search and working values').replace('Search intervals and selected values','Development search intervals and working values').replace(' & Selected & ',' & Working & '))
# Common physical / task fields and full fields excluded from the search.
excluded=set(tuning['spaces']['full'])-{'t2_ratio'}
lines=[r'\begin{longtable}{lr}',r'\caption{Configuration fields not listed as search variables in the full-model table. Units are nondimensional; \texttt{noise} is overridden by task. The \texttt{frozen\_tone} field is unused by the exact freeze-at-damage control.}\\',r'\toprule Configuration key & Value \\\midrule\endfirsthead',r'\toprule Configuration key & Value \\\midrule\endhead',r'\bottomrule\endfoot']
for key,value in selected['full']['config'].items():
 if key in excluded:continue
 if isinstance(value,float):value=f'{value:.9g}'
 lines.append(r'\texttt{'+esc(key)+'} & '+esc(str(value))+r' \\')
lines.append(r'\end{longtable}');(TAB/'fixed_parameters.tex').write_text('\n'.join(lines)+'\n')
threshold=rows('threshold_sensitivity.csv')
lines=[r'\begin{longtable}{llrrr}',r'\caption{Range across all six recovery definitions: $F_*=0.90,0.95,1.00$ and $\tau_*=2,5$. The range refers to means over the same paired seeds, not variability among individual trials.}\\',r'\toprule Task & Controller & Min. mean $T_{\rm rec}$ & Max. mean $T_{\rm rec}$ & Healed range \\\midrule\endfirsthead',r'\toprule Task & Controller & Min. mean $T_{\rm rec}$ & Max. mean $T_{\rm rec}$ & Healed range \\\midrule\endhead',r'\bottomrule\endfoot']
for case in CASES:
 for mode in EXT+INT[1:]:
  g=[r for r in threshold if r['case']==case and r['controller']==mode];t=[float(r['mean_restricted_recovery_time']) for r in g];h=[int(r['healed_runs']) for r in g]
  lines.append(f'{case.capitalize()} & {LABEL[mode]} & {num(min(t))} & {num(max(t))} & {min(h)} to {max(h)}/{len(P["seeds"])}'+r' \\')
lines.append(r'\end{longtable}');(TAB/'thresholds.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{longtable}{llrr}',r'\caption{Mean progress $U$ and safety-guard intervention fraction $G$, with sample SD over seeds. $G$ is the fraction of agent steps whose requested motion is reduced by the guard. All 2,400 runs satisfy the stored clearance criterion.}\\',r'\toprule Task & Controller & $U$ & $G$ \\\midrule\endfirsthead',r'\toprule Task & Controller & $U$ & $G$ \\\midrule\endhead',r'\bottomrule\endfoot']
for case in CASES:
 for mode in EXT+INT[1:]:
  r=A[case,mode]
  lines.append(f'{case.capitalize()} & {LABEL[mode]} & '+ms(r,'progress_fraction',4)+' & '+ms(r,'guard_fraction_mean',6)+r' \\')
lines.append(r'\end{longtable}');(TAB/'guard.tex').write_text(('\n'.join(lines)+'\n').replace('2,400',f"{len(P['main_cases'])*len(P['controllers'])*len(P['seeds']):,}"))

plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):fig.savefig(FIG/name,bbox_inches='tight',pad_inches=.035);plt.close(fig)
# Actual split followed by healing, chosen first seed; no selection on trajectory outcome.
d=load_run('moderate','full',P['seeds'][0])
fig,axes=plt.subplots(1,4,figsize=(7.2,2.2),layout='constrained')
for ax,t,letter in zip(axes,[0,.8,1.6,4.],list('abcd')):
 i=np.argmin(abs(d['snapshot_time']-t));xy=d['positions'][i].copy();xy[:,1]-=xy[:,1].mean();theta=d['headings'][i]
 delta=xy[:,None]-xy[None,:];distance=np.linalg.norm(delta,axis=-1)
 ii,jj=np.where(np.triu((distance<3.2)&(distance>0),1))
 ax.add_collection(LineCollection(np.stack([xy[ii],xy[jj]],axis=1),colors='#d5dce0',lw=.35,zorder=0))
 ax.quiver(xy[:,0],xy[:,1],np.cos(theta),np.sin(theta),color=np.where(d['groups']==0,'#16708d','#cd7626'),angles='xy',scale_units='xy',scale=2,width=.008)
 ti=np.argmin(abs(d['time']-t));ax.set_title(f'({letter}) $t={t:g}$\n$F={d["cluster"][ti]:.2f}$, $P={d["polar"][ti]:.2f}$')
 ax.set(xlim=(-6,6),ylim=(-5,5),xlabel='$x$');ax.set_aspect('equal');ax.set_xticks([-4,0,4]);ax.set_yticks([-4,0,4])
axes[0].set_ylabel('$y-\\bar y$');save(fig,'fig1_split_recovery.pdf')
# External comparison first: time evolution, means and seed SD.
fig,axes=plt.subplots(3,2,figsize=(7.2,6.1),layout='constrained')
for i,case in enumerate(CASES):
 for mode in EXT:
  values={k:[] for k in ['cluster','polar']}
  for seed in P['seeds']:
   z=load_run(case,mode,seed)
   times=z['time']
   for k in values:values[k].append(z[k])
  for j,k in enumerate(values):
   v=np.asarray(values[k]);mean=v.mean(0);sd=v.std(0,ddof=1);ax=axes[i,j]
   ax.plot(times,mean,color=COL[mode],lw=1.3,label=PLABEL[mode]);ax.fill_between(times,np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=COL[mode],alpha=.12,lw=0)
 for j,ax in enumerate(axes[i]):
  ax.axhline(.9,color='#777777',ls=':',lw=.65);ax.set(xlim=(0,45 if case=='obstacle' else 6),ylim=(0,1.03),xlabel='Time after imposed split',ylabel='$F$' if j==0 else '$P$');ax.grid(alpha=.16)
  ax.set_title(f'({chr(97+2*i+j)}) {case.capitalize()}')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='outside lower center',frameon=False)
save(fig,'fig2_external_recovery.pdf')
# Within-framework causal contrasts, relative effects and simultaneous intervals.
fig,axes=plt.subplots(1,2,figsize=(7.2,3.6),layout='constrained')
for ax,ep,title in zip(axes,['restricted_recovery_time','joint_recovery_deficit'],['(a) Recovery time','(b) Joint deficit']):
 for i,case in enumerate(CASES):
  for j,m in enumerate(['fast','frozen','scalar_matched']):
   r=C[case,m,ep];y=3*i+j
   mid,lo,hi=[float(r[k]) for k in ['improvement_pct','improvement_family_low','improvement_family_high']]
   ax.plot([lo,hi],[y,y],color=COL[m],lw=1.5)
   sig=float(r['permutation_p_holm'])<.05;ax.plot(mid,y,'o',ms=4.5,mec=COL[m],mfc=COL[m] if sig else 'white')
 ax.axvline(0,color='#666666',ls=':',lw=.8);ax.set_title(title);ax.set_yticks(range(9),[c.capitalize()+' / '+PLABEL[m] for c in CASES for m in ['fast','frozen','scalar_matched']],fontsize=7.5)
 ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.set_xlabel('Reduction with full (%)')
 for y in [2.5,5.5]:ax.axhline(y,color='#dddddd',lw=.6)
save(fig,'fig3_feedback_effects.pdf')
runpy.run_path(str(ROOT/'plotting/build_architecture.py'), run_name='__main__')
manifest={'input_protocol_sha256':hashlib.sha256((DATA/'protocol.json').read_bytes()).hexdigest(),'source':os.path.relpath(DATA,ROOT),'figures':[p.name for p in sorted(FIG.glob('*.pdf'))],'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tables_are_generated_from_saved_data':True}
manifest['architecture_generator_sha256']=hashlib.sha256((ROOT/'plotting/build_architecture.py').read_bytes()).hexdigest()
manifest['architecture_is_schematic']=True
(HERE/'asset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built manuscript assets from audited saved outcomes.')
