from pathlib import Path
import csv,json,os
from decimal import Decimal,ROUND_HALF_UP
ROOT=Path(__file__).resolve().parents[1]
spec=json.loads(Path(os.environ.get('HEALING_SELECTION',str(ROOT/'configurations/working.json'))).read_text())
OUT=Path(os.environ.get('HEALING_ASSET_OUT',str(ROOT/'results/assets')))/'tables'
OUT.mkdir(parents=True,exist_ok=True)
labels={'olfati_saber':'Olfati-Saber','couzin':'Couzin','vasarhelyi':r"V\'{a}s\'{a}rhelyi"}
lines=[r'\begin{tabular}{llrlr}',r'\toprule Model & Parameter & Value & Parameter & Value \\\midrule']
for mode,label in labels.items():
 items=list(spec[mode]['parameters'].items());pairs=[items[i:i+2] for i in range(0,len(items),2)]
 for i,pair in enumerate(pairs):
  a,v=pair[0];b,w=pair[1] if len(pair)>1 else ('','')
  esc=lambda x:r'\texttt{'+x.replace('_',r'\_')+'}' if x else ''
  lines.append((label if i==0 else '')+' & '+esc(a)+f' & {v:g} & '+esc(b)+' & '+(f'{w:g}' if w!='' else '')+r' \\')
 if mode!='vasarhelyi':lines.append(r'\midrule')
lines += [r'\bottomrule',r'\end{tabular}'];param='\n'.join(lines)+'\n'
(OUT/'baseline_parameters.tex').write_text(param)
rows=list(csv.DictReader((Path(os.environ.get('HEALING_GENERALIZATION',str(ROOT/'results/generalization')))/'aggregate.csv').open()))
def ms(r,k):return '$'+r'\,\pm\,'.join(str(Decimal(f'{float(r[k+t]):.10f}').quantize(Decimal('.001'),rounding=ROUND_HALF_UP)) for t in ['_mean','_sd'])+'$'
lab={'reference':'Reference','population_24':'$N=24$','population_96':'$N=96$','dt_025':r'$\Delta t=0.025$','dt_0125':r'$\Delta t=0.0125$','centering_half':r'$k_x=0.0175$','centering_zero':r'$k_x=0$'}
lines=[r'\begin{longtable}{llrrr}',r'\caption{Full-controller sensitivity: mean $\pm$ sample SD over 30 paired seeds per row. The complete all-model table is supplied as \texttt{data/rounded/generalization/aggregate.csv}. Lower $T_{\rm rec}$ and $J$ are better.}\label{tab:generalization}\\',r'\toprule Setting & Task & $T_{\rm rec}$ & $J$ & Healed \\\midrule\endfirsthead',r'\toprule Setting & Task & $T_{\rm rec}$ & $J$ & Healed \\\midrule\endhead',r'\bottomrule\endfoot']
for r in rows:
 if r['controller']=='full':lines.append((lab[r['setting']] if r['case']=='moderate' else '')+' & '+r['case'].capitalize()+' & '+ms(r,'restricted_recovery_time')+' & '+ms(r,'joint_recovery_deficit')+' & '+r['healed_runs']+'/30'+r' \\'+('*' if r['case']!='obstacle' else ''))
lines.append(r'\end{longtable}');full='\n'.join(lines)+'\n'
(OUT/'generalization_full.tex').write_text(full)
