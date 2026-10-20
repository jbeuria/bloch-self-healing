"""PDF-only supplementary plots and tables from fixed-parameter saved results."""
from pathlib import Path
import csv,json,hashlib,os
from decimal import Decimal,ROUND_HALF_UP
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('HEALING_GENERALIZATION',str(ROOT/'results/generalization')));OUT=Path(os.environ.get('HEALING_ASSET_OUT',str(ROOT/'manuscript')))
FIG=OUT/'figures';TAB=OUT/'tables';FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
P=json.loads((DATA/'protocol.json').read_text())
assert hashlib.sha256((DATA/'protocol.json').read_bytes()).hexdigest()==(DATA/'protocol.sha256').read_text().strip()
A={(r['setting'],r['case'],r['controller']):r for r in csv.DictReader((DATA/'aggregate.csv').open())}
LABEL={'full':'Full','fast':'No feedback','frozen':'Frozen self','scalar_matched':'Matched scalar','olfati_saber':'Olfati-Saber','couzin':'Couzin','vasarhelyi':r"V\'{a}s\'{a}rhelyi"}
PLOT_LABEL={**LABEL,'olfati_saber':'Olfati-Saber','vasarhelyi':'Vásárhelyi'}
COLOR={'full':'#086f83','olfati_saber':'#d35d12','couzin':'#8053a0','vasarhelyi':'#618f28','fast':'#30363b','frozen':'#ab7b32','scalar_matched':'#ad537e'}
SLABEL={'reference':r'Reference: $N=48$, $\Delta t=0.05$, $k_x=0.035$',
        'population_24':r'$N=24$','population_96':r'$N=96$',
        'dt_025':r'$\Delta t=0.025$','dt_0125':r'$\Delta t=0.0125$',
        'centering_half':r'$k_x=0.0175$ (half)','centering_zero':r'$k_x=0$ (removed)'}

def ms(row,metric):
    v=[str(Decimal(f"{float(row[metric+k]):.10f}").quantize(Decimal(".001"),rounding=ROUND_HALF_UP)) for k in ("_mean","_sd")]
    return "$"+v[0]+r"\,\pm\,"+v[1]+"$"
lines=[r'\begin{longtable}{llrrr}',r'\caption{Complete fixed-parameter supplementary results: mean $\pm$ sample SD over 30 fresh paired seeds per row. $T_{\rm rec}$ and $J$ are lower-is-better; unconfirmed first recoveries contribute $H=45$. The seven settings vary one factor at a time; the fresh reference is shared by all three sensitivity axes. All controllers retain their primary-study working parameters.}\label{tab:generalization}\\',r'\toprule Task & Controller & $T_{\rm rec}$ & $J$ & Healed \\\midrule\endfirsthead',r'\toprule Task & Controller & $T_{\rm rec}$ & $J$ & Healed \\\midrule\endhead',r'\bottomrule\endfoot']
for setting in P['settings']:
 lines.append(r'\multicolumn{5}{l}{\textbf{'+SLABEL[setting]+r'}}\\*')
 for case in P['tasks']:
  for mi,mode in enumerate(P['controllers']):
   r=A[setting,case,mode]
   lines.append((case.capitalize() if mi==0 else '')+' & '+LABEL[mode]+' & '+ms(r,'restricted_recovery_time')+' & '+ms(r,'joint_recovery_deficit')+f" & {r['healed_runs']}/{r['n']}"+r' \\'+('*' if mi<len(P['controllers'])-1 else ''))
  lines.append(r'\addlinespace[2pt]')
lines.append(r'\end{longtable}');(TAB/'generalization_all.tex').write_text('\n'.join(lines)+'\n')

plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
axes_specs=[(['population_24','reference','population_96'],['24','48','96'],'Population $N$'),
            (['reference','dt_025','dt_0125'],['0.05','0.025','0.0125'],r'Time step $\Delta t$'),
            (['reference','centering_half','centering_zero'],['1','0.5','0'],'Centering multiplier')]
fig,axes=plt.subplots(3,3,figsize=(7.2,6.1),layout='constrained')
for ci,case in enumerate(P['tasks']):
 for j,(settings,ticks,xlabel) in enumerate(axes_specs):
  ax=axes[ci,j]
  for mi,mode in enumerate(P['controllers']):
   x=np.arange(3)+(mi-3)*.035
   y=[float(A[s,case,mode]['joint_recovery_deficit_mean']) for s in settings]
   sd=[float(A[s,case,mode]['joint_recovery_deficit_sd']) for s in settings]
   ax.errorbar(x,y,yerr=np.array([np.minimum(sd,y),sd]),color=COLOR[mode],lw=1.1 if mode=='full' else .8,ls='-' if mi<4 else '--',marker='o' if mode=='full' else '.',ms=3.2 if mode=='full' else 2.5,capsize=1.1,elinewidth=.5,label=PLOT_LABEL[mode])
  ax.set_xticks(range(3),ticks);ax.set_xlabel(xlabel);ax.set_title(f'({chr(97+ci*3+j)}) {case.capitalize()}');ax.grid(alpha=.16)
  if j==0:ax.set_ylabel('Joint deficit $J$')
  ax.set_ylim(bottom=0)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='outside lower center',frameon=False)
fig.savefig(FIG/'figS1_generalization.pdf',bbox_inches='tight',pad_inches=.04);plt.close(fig)
(OUT/'generalization_asset_manifest.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256((DATA/'protocol.json').read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source':os.path.relpath(DATA,ROOT),'outputs':['figures/figS1_generalization.pdf','tables/generalization_all.tex'],'simulation_called':False},indent=2)+'\n')
print('Built supplementary PDF plot and complete mean/SD table.')
