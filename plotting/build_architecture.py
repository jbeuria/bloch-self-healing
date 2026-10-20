"""Draw the implemented observation, memory and action map as a vector PDF only."""
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MPath

HERE = Path(os.environ.get('HEALING_ASSET_OUT',str(Path(__file__).resolve().parents[1]/'manuscript')))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 17,
                     'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(16.8, 7.1))
fig.subplots_adjust(left=.005, right=.995, bottom=.01, top=.99)
ax.set(xlim=(-.05, 16.75), ylim=(-.2, 6.95))
ax.axis('off')
ink, feedback = '#263943', '#9b631a'

def box(x, y, w, h, title, lines, color, title_size=18):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=0.10',
                              linewidth=1.4,edgecolor=ink,facecolor=color,zorder=2))
    ax.text(x+w/2,y+h-.23,title,ha='center',va='top',weight='bold',
            fontsize=title_size,linespacing=1.2,zorder=3)
    start=y+h-(.96 if '\n' in title else .7)
    for k,line in enumerate(lines):
        ax.text(x+w/2,start-.39*k,line,ha='center',va='top',fontsize=16.5,zorder=3)

def arrow(points, dashed=False, color=ink, width=1.5):
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=17,
                                linewidth=width,color=color,
                                linestyle=(0,(4,3)) if dashed else '-',zorder=1))

def label(x,y,text,color=ink,size=15,va='center'):
    ax.text(x,y,text,ha='center',va=va,color=color,fontsize=size,
            bbox=dict(facecolor='white',edgecolor='none',pad=2),zorder=4)

box(.2,3.3,3.25,2.4,'Local observations',[
    r'$A_{ij},\;\boldsymbol{x}_j-\boldsymbol{x}_i$',
    r'$\boldsymbol{e}_i\!\cdot\!\boldsymbol{e}_j$',
    'obstacle clearance',
    'shared navigation cues'],'#e8f1eb')
box(4.2,3.3,3.25,2.4,'Bloch channels',[
    r'$\dot{\boldsymbol{m}}=2\boldsymbol{h}\times\boldsymbol{m}$',
    r'$\quad-\Lambda(\boldsymbol{m}-\tanh H\,\boldsymbol{e}_z)$',
    r'$H=g_a\boldsymbol{e}_i\!\cdot\!\boldsymbol{e}_j+\kappa s_i$',
    r'$\|\boldsymbol{m}_{ij}\|\leq 1$'],'#fff1d9')
box(8.2,3.3,3.25,2.4,'Steering map',[
    r'$k_a\boldsymbol{a}_i+k_c\boldsymbol{c}_i$',
    r'$+\,b(s_i)(k_r\boldsymbol{r}_i+k_o\boldsymbol{o}_i)$',
    r'$+\,\boldsymbol{n}(\boldsymbol{x}_i)$',
    'desired heading'],'#e4f1f6')
box(12.2,3.3,3.25,2.4,'Motion interface',[
    'velocity relaxation',
    'noise; speed / turn caps',
    'acceleration cap; guard',
    r'$\boldsymbol{x}_i^{n+1},\boldsymbol{v}_i^{n+1},\theta_i^{n+1}$'],'#ecedf1')
box(4.2,.9,3.25,1.45,'Slow self state',[
    r'$\dot s_i=\gamma_s[s_{\rm base}$',
    r'$+\tanh(\lambda_{\rm fb}M_i)-s_i]$'],'#efe8f5')
box(12.2,.9,3.25,1.45,'Evaluation only',[
    r'$F,\;P,\;T_{\rm rec},\;J$',
    'progress; healing success'],'#f7e9e7')

# Direct perceptual and action paths.
arrow([(3.49,4.45),(4.15,4.45)])
arrow([(7.49,4.45),(8.15,4.45)])
label(7.82,4.91,r'$w_{ij}$',size=18)
arrow([(11.49,4.45),(12.15,4.45)])
label(11.82,4.91,r'$\boldsymbol{v}^{\rm des}$',size=17)

# Geometry and shared task cues also reach steering, independently of memory.
arrow([(1.82,5.75),(1.82,6.38),(9.82,6.38),(9.82,5.76)])
label(5.82,6.38,'local geometry and shared navigation cues',size=16)

# Updated perceptual mean drives self; the two dashed paths carry the old s.
arrow([(6.77,3.25),(6.77,2.4)])
label(6.77,2.83,r'$M_i$',size=18)
arrow([(4.83,2.4),(4.83,3.25)],dashed=True,color=feedback)
label(4.83,2.83,r'$\kappa s_i$',color=feedback,size=18)
arrow([(7.5,1.61),(9.82,1.61),(9.82,3.25)],dashed=True,color=feedback)
label(9.82,2.55,r'$b(s_i)$',color=feedback,size=18)

# Physical observations close the sensorimotor loop; evaluation is a sink.
arrow([(13.82,3.25),(13.82,2.4)])
label(13.82,2.83,'record',size=15)
arrow([(15.5,4.45),(16.3,4.45),(16.3,.22),(.0,.22),(.0,3.55),(.15,3.55)])
label(8.2,.22,'physical state at the next step',size=16)
ax.text(.28,1.88,'Finite-range sensing\nNo state of distant groups',ha='left',va='center',
        fontsize=15.5,color=ink,linespacing=1.5,
        bbox=dict(facecolor='white',edgecolor='none',pad=3),zorder=4)

out=HERE/'figures'/'architecture.pdf'
out.parent.mkdir(exist_ok=True)
fig.savefig(out,bbox_inches='tight',pad_inches=.08,
            metadata={'Title':'Observation-memory-action architecture',
                      'Author':'Jyotiranjan Beuria',
                      'Subject':'Implemented 2D bounded-memory flock controller; evaluation is not feedback'})
plt.close(fig)
print(out)
