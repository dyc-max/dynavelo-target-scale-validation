"""Render frozen manuscript results; no model inference or statistical fitting."""
from pathlib import Path
import sys,json,shutil,zipfile
import numpy as np,pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath
import argparse
parser=argparse.ArgumentParser(description='Render figures from frozen source tables; no model or differential analysis.')
parser.add_argument('--source-data',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
DATA=args.source_data.resolve(); OUT=args.output.resolve()
if OUT.exists():raise SystemExit('Choose a new output directory.')
FIG=OUT/'Figures'; QA=OUT/'Quality_Checks'
for d in [FIG,QA]: d.mkdir(parents=True,exist_ok=True)

RAW='#697b8c'; L1='#168879'; PCT='#8a5ba6'; DARK='#243b4b'; ORANGE='#c56c2b'
COLORS=[RAW,L1,PCT]; LABELS=['Raw','Column L1','Column percentile']
mpl.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'savefig.dpi':300})
def title(ax,letter,s):
    ax.set_title(s,loc='left',pad=14,fontweight='bold')
    ax.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-28,14),textcoords='offset points',weight='bold',fontsize=12)
def save(fig,n):
    fig.canvas.draw()
    # Local authoring QA dependency omitted from the portable rendering entry point.
    fig.savefig(FIG/f'Figure_{n}.pdf',facecolor='white')
    fig.savefig(FIG/f'Figure_{n}.svg',facecolor='white')
    fig.savefig(FIG/f'Figure_{n}.png',dpi=600,facecolor='white')
    plt.close(fig)

# Figure 1: four steps with distinct scientific roles.
fig,axes=plt.subplots(2,2,figsize=(7.2,5.9))
fig.subplots_adjust(left=.08,right=.98,top=.91,bottom=.05,wspace=.22,hspace=.38)
blocks=[('a','Discovery and screening',['Human bone-marrow CD34+ HSPCs','Day 7; GSE284047 / 8489-MV-1','6,035 cells × 3,006 RNA features','1,191 inputs · 3 seeds · 2 sampling sets','20 retained candidate regulators']),('b','Local sensitivity readout',['Input change → target kinetic response','Signed sensitivity: directional hypothesis','Absolute magnitude: ranking strength','Full-population evaluation: 1,884 cells','Common score-comparison matrix: 192 cells']),('c','Target-scale diagnosis',['Top-50 coverage and target occupancy','Expression variance and column totals','Raw magnitude → column L1 / percentile','Same inputs, targets and seed aggregation','Absolute and relative rankings retained']),('d','Complementary validation',['Algorithmic: GRNBoost2 rank concordance','All 3,005 non-self targets per candidate','Functional: independent perturbation data','CEBPA–CSF3R · CEBPA–MPO · XBP1–HSPA5','Interpret responses within cellular context'])]
for ax,(letter,head,lines) in zip(axes.flat,blocks):
    ax.set_axis_off(); title(ax,letter,head)
    ax.add_patch(FancyBboxPatch((0,.02),1,.9,boxstyle='round,pad=0.01,rounding_size=0.025',linewidth=.8,edgecolor='#c4ced6',facecolor='#f5f7f9',transform=ax.transAxes))
    if letter!='c':
        for k,line in enumerate(lines): ax.text(.045,.79-k*.16,line,transform=ax.transAxes,va='center',fontsize=8.4,color=DARK)
    else:
        ax.text(.045,.80,'1,191 inputs × 3,006 target columns',fontsize=8.7,color=DARK,transform=ax.transAxes)
        ax.text(.045,.65,'Normalize within each target column',fontsize=8.0,color=DARK,transform=ax.transAxes)
        for block,x0 in [('Raw',.06),('Normalized',.60)]:
            for row in range(4):
                for col in range(5):
                    # Conceptual matrix glyphs: no measured data are encoded.
                    color=RAW if block=='Raw' else PCT
                    alpha=[.3,1,.6,.8,.4][col]*[.3,.5,.7,.9][row] if block=='Raw' else [.3,.5,.7,.9][row]
                    ax.add_patch(Rectangle((x0+col*.061,.22+row*.065),.052,.052,facecolor=color,alpha=alpha,edgecolor='none',transform=ax.transAxes))
            ax.text(x0+.14,.13,block,ha='center',fontsize=8,transform=ax.transAxes)
        ax.annotate('',xy=(.57,.35),xytext=(.39,.35),xycoords='axes fraction',arrowprops=dict(arrowstyle='-|>',lw=1.1,color=DARK))
        ax.text(.48,.47,'L1 / rank',ha='center',fontsize=7.6,transform=ax.transAxes)
        ax.text(.48,.045,'Conceptual column-normalization schematic',ha='center',fontsize=7.2,color=DARK,transform=ax.transAxes)
# Workflow arrows connect step boxes through the inter-panel gutters.
pos=[a.get_position() for a in axes.flat]
for i,j in [(0,1),(2,3)]:
    a,b=pos[i],pos[j]
    fig.add_artist(FancyArrowPatch((a.x1+.006,a.y0+.45*a.height),(b.x0-.006,b.y0+.45*b.height),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=11,lw=1.2,color=DARK))
a,b=pos[1],pos[2]
mid=(a.y0+b.y1)/2
connector=MplPath([(a.x0+.04,a.y0-.01),(a.x0+.04,mid),(b.x1-.015,mid),(b.x1-.015,b.y1+.015)])
fig.add_artist(FancyArrowPatch(path=connector,transform=fig.transFigure,arrowstyle='-|>',mutation_scale=11,lw=1.2,color=DARK))
save(fig,1)

# Source data are selections of existing values, not recomputed statistics.
diag=pd.read_csv(DATA/'S2_Target_scale_profiles.csv',float_precision='round_trip')
shared=diag.loc[diag['Raw occurrence'].eq(20)].copy()
assert len(shared)==13
fig,axes=plt.subplots(2,2,figsize=(7.2,6.1))
fig.subplots_adjust(left=.13,right=.97,top=.92,bottom=.10,wspace=.48,hspace=.55)
for ax,values,letter,head,ylabel,ylim in [(axes[0,0],[87,532,620],'a','Distinct target coverage','Distinct targets',(0,710)),(axes[0,1],[13,0,0],'b','Universal target occupancy','Targets in all 20 lists',(0,16))]:
    title(ax,letter,head);ax.bar(range(3),values,color=COLORS,width=.55)
    for x,v in enumerate(values):ax.text(x,v+ylim[1]*.04,str(v),ha='center',fontsize=10,weight='bold')
    ax.set(xticks=range(3),xticklabels=['Raw','L1','Percentile'],ylabel=ylabel,ylim=ylim)
ax=axes[1,0];title(ax,'c','Shared targets across scores')
ax.imshow(shared[['Raw occurrence','L1 occurrence','Percentile occurrence']].to_numpy(),cmap=mpl.colors.ListedColormap(['#f1f3f6',RAW]),vmin=0,vmax=20,aspect='auto')
ax.set_xticks(range(3),['Raw','L1','Percentile']);ax.set_yticks(range(13),shared.Target,fontsize=7.5)
for y in range(13):
    for x,v in enumerate([20,0,0]):ax.text(x,y,str(v),ha='center',va='center',fontsize=7.3,color='white' if v else '#55616d')
ax.tick_params(length=0)
ax=axes[1,1];title(ax,'d','Variance and column scale')
ax.scatter(shared['Expression variance'],shared['Sensitivity column total'],s=24,c=RAW,alpha=.85)
mpo=diag.loc[diag.Target.eq('MPO')].iloc[0]
ax.scatter([mpo['Expression variance']],[mpo['Sensitivity column total']],s=65,c=PCT,marker='D')
for gene,offset in [('MPO',(-28,11)),('SRGN',(5,-1)),('CST3',(5,0))]:
    r=diag.loc[diag.Target.eq(gene)].iloc[0];ax.annotate(gene,(r['Expression variance'],r['Sensitivity column total']),xytext=offset,textcoords='offset points',fontsize=8,color=PCT if gene=='MPO' else DARK)
ax.set(xlabel='Expression variance',ylabel='Sensitivity-column total',xlim=(0,1.85),ylim=(0,18))
save(fig,2)

wide=pd.read_csv(DATA/'S1_Candidate_rank_concordance.csv',index_col='Candidate',float_precision='round_trip')
fig=plt.figure(figsize=(7.2,6.2));gs=fig.add_gridspec(2,2,left=.13,right=.97,top=.92,bottom=.15,wspace=.45,hspace=.68)
ax=fig.add_subplot(gs[:,0]);title(ax,'a','All 20 candidate regulators')
for y,row in enumerate(wide.to_numpy()):ax.plot(row,[y]*3,c='#ccd2d8',lw=1,zorder=1)
for k,c in enumerate(COLORS):ax.scatter(wide.iloc[:,k],np.arange(20),s=26,color=c,label=LABELS[k],zorder=2,marker=['o','s','D'][k])
ax.set(yticks=range(20),yticklabels=wide.index,xlabel='Spearman ρ with GRNBoost2',xlim=(.25,.61),ylim=(19.7,-.7))
handles,legend_labels=ax.get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.52,.008),fontsize=8,ncol=3,handletextpad=.4)
ax=fig.add_subplot(gs[0,1]);title(ax,'b','Candidate-level distribution')
medians=[.4832863362097343,.39329683822968675,.38566625206208965]
for k,c in enumerate(COLORS):
    # Deterministic offsets expose all frozen values; no stochastic sampling.
    xx=k+np.linspace(-.11,.11,20)
    ax.scatter(xx,wide.iloc[:,k],color=c,s=18,alpha=.85)
    ax.plot([k-.20,k+.20],[medians[k]]*2,c='black',lw=1.5)
    ax.text(k,.605,f'{medians[k]:.4f}',ha='center',fontsize=8)
ax.set(xticks=range(3),xticklabels=['Raw','L1','Percentile'],ylabel='Spearman ρ',ylim=(.25,.63))
ax=fig.add_subplot(gs[1,1]);title(ax,'c','CEBPA–MPO rank recovery');ax.axis('off')
for k,(label,state,score) in enumerate(zip(['Raw magnitude','Column L1','Column percentile'],['Outside top 50','Outside top 50','Top-score tie: 56 targets'],['0.0022181863','0.0040286523','0.9995801847'])):
    yy=.91-k*.30;ax.text(0,yy,label,color=COLORS[k],weight='bold',fontsize=9)
    ax.text(0,yy-.105,state,fontsize=8.6);ax.text(0,yy-.20,'Score: '+score,fontsize=8,color='#55616d')
ax.text(0,-.055,'Index order 49; included in exact top 50',fontsize=7.5,color=PCT)
ax.text(0,-.17,'Signed median: +0.0010788117',fontsize=8.2,color=DARK)
ax.text(0,-.29,'Seed signs: − / + / + (2 positive, 1 negative)',fontsize=7.5,color=PCT)
save(fig,3)

effects=pd.read_csv(DATA/'Figure_4_Effects.csv',float_precision='round_trip')
fig,axes=plt.subplots(2,2,figsize=(7.2,5.3))
fig.subplots_adjust(left=.12,right=.97,top=.88,bottom=.13,wspace=.50,hspace=.85)
groups=[('a','CEBPA induction → CSF3R',effects.iloc[[0]],['K562 · 48 h'],(-1,6)),('b','Cebpa deletion → Mpo',effects.iloc[[1,2]],['HSPCs · unstimulated','HSPCs · FLT3L 4 h'],(-3.5,1)),('c','Xbp1 deletion → Hspa5',effects.iloc[[3,4]],['Splenic CD8α DC','Tumor cDC1'],(-.15,.7)),('d','XBP1 CRISPRi → HSPA5',effects.iloc[[5]],['Human CD4 T cells · 48 h'],(-1.2,.4))]
for ax,(letter,head,rows,labels,xlim) in zip(axes.flat,groups):
    title(ax,letter,head)
    for i,(_,r) in enumerate(rows.iterrows()):
        color=ORANGE if letter=='d' else L1
        ax.vlines(0,i-.22,i+.22,color='#bac2ca',lw=.9,ls='--',zorder=0)
        if pd.notna(r.Low):ax.errorbar(r.Effect,i,xerr=[[r.Effect-r.Low],[r.High-r.Effect]],fmt='o',ms=6,color=color,capsize=3,lw=1.5)
        else:ax.plot(r.Effect,i,'o',ms=7,markerfacecolor='white',markeredgecolor=color,markeredgewidth=1.5)
        ax.text(xlim[0]+.015*(xlim[1]-xlim[0]),i+.36,labels[i],fontsize=8.5)
        ax.text(xlim[1],i+.36,f'{r.Effect:+.3f}',fontsize=8.5,ha='right',weight='bold',color=color)
    ax.set(xlim=xlim,ylim=(-.65,len(rows)-.25),yticks=[],xlabel='log2 change: perturbation − control')
    ax.spines['left'].set_visible(False)
fig.text(.12,.025,'Filled points: estimate with 95% model interval.  Open points: interval unavailable.',fontsize=8,color='#55616d')
save(fig,4)

