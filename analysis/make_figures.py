#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera as figuras do artigo em figs/: diagramas de diferença crítica (cd_ARI, cd_AMI), dispersão GDT_m vs.
baselines (scatter_ARI), mapas de calor de ganho (heatmap_gain_ARI*, incluindo a versão combinada) e tempo (runtime)."""
import argparse, os, sys
_p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument('--input', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'paper', 'benchmark_long.csv'),
                help='CSV longo gerado por benchmark_hclust.py (padrão: resultados do artigo)')
_p.add_argument('--outdir', default='outputs', help='pasta de saída (padrão: outputs)')
ARGS = _p.parse_args()
INPUT = os.path.abspath(ARGS.input)
os.makedirs(ARGS.outdir, exist_ok=True); os.chdir(ARGS.outdir); os.makedirs('figs', exist_ok=True)

# ======================================================================
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.edgecolor':'#898781','axes.labelcolor':'#0b0b0b',
  'xtick.color':'#52514e','ytick.color':'#52514e','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
df=pd.read_csv(INPUT); ds=list(dict.fromkeys(df.dataset))
W=['euclid','tangent','zmp','gdt','gdt_full']; LK=['ward','complete','average']
LBL={'euclid':'Euclid','tangent':'Tangent','zmp':'ZMP','gdt':r'GDT$_d$','gdt_full':r'GDT$_m$'}
LC={'ward':'#2a78d6','complete':'#eb6834','average':'#1baf7a'}; LM={'ward':'o','complete':'s','average':'^'}
INK='#0b0b0b'; INK2='#52514e'; MUTED='#898781'
def piv(metric,lk): return df[df.linkage==lk].pivot(index='dataset',columns='weight',values=metric).loc[ds,W]
# ---------- 1. CD diagrams ----------
def cd_plot(ax, ranks, names, cd, title):
    k=len(names); lo,hi=1,k
    ax.set_xlim(lo-0.3,hi+0.3); ax.set_ylim(0.3,1.06); ax.axis('off')
    ax.hlines(0.78,lo,hi,color=INK,lw=0.8)
    for t in range(lo,hi+1):
        ax.vlines(t,0.78,0.82,color=INK,lw=0.8); ax.text(t,0.86,str(t),ha='center',va='bottom',fontsize=8,color=INK2)
    order=np.argsort(ranks); half=(k+1)//2
    for idx,i in enumerate(order):
        r=ranks[i]; left = idx<half
        y=0.62-0.13*(idx if left else k-1-idx)
        xt = lo-0.25 if left else hi+0.25
        ax.plot([r,r,xt],[0.78,y,y],color=INK,lw=0.7)
        bold = names[i]==r'GDT$_m$'
        ax.text(xt+(-0.03 if left else 0.03), y, f'{names[i]} ({r:.2f})', ha='right' if left else 'left', va='center',
                fontsize=8, color=INK, weight='bold' if bold else 'normal')
    # cliques: maximal groups with range < cd
    srt=np.sort(ranks); cl=[]
    for a in range(k):
        b=a
        while b+1<k and srt[b+1]-srt[a]<cd: b+=1
        if b>a and not any(c[0]<=a and c[1]>=b for c in cl): cl.append((a,b))
    for n_,(a,b) in enumerate(cl):
        y=0.72-0.045*n_; ax.hlines(y,srt[a]-0.03,srt[b]+0.03,color=INK,lw=2.2)
    ax.hlines(0.97,lo,lo+cd,color=INK,lw=0.8); ax.vlines([lo,lo+cd],0.95,0.99,color=INK,lw=0.8)
    ax.text(lo+cd/2,1.0,f'CD = {cd:.2f}',ha='center',va='bottom',fontsize=7.5,color=INK2)
    ax.set_title(title,fontsize=9.5,color=INK,pad=14)
q=2.728; N=len(ds); k=5; CD=q*np.sqrt(k*(k+1)/(6*N))
for metric in ['ARI','AMI']:
    fig,axs=plt.subplots(1,3,figsize=(10.5,1.9))
    for ax,lk in zip(axs,LK):
        t=piv(metric,lk); R=np.vstack([rankdata(-np.round(r,3)) for r in t.values]).mean(0)
        cd_plot(ax,R,[LBL[w] for w in W],CD,lk.capitalize()+' linkage')
    plt.tight_layout(); plt.savefig(f'figs/cd_{metric}.pdf',bbox_inches='tight'); plt.savefig(f'figs/cd_{metric}.png',dpi=200,bbox_inches='tight'); plt.close()
print('CD',CD)
# ---------- 2. scatter vs baselines ----------
fig,axs=plt.subplots(1,3,figsize=(10.5,3.6),sharey=True)
for ax,b in zip(axs,['euclid','tangent','zmp']):
    wins=0;tot=0
    for lk in LK:
        t=piv('ARI',lk)
        ax.scatter(t[b],t.gdt_full,s=26,marker=LM[lk],color=LC[lk],edgecolor='#fcfcfb',linewidth=0.6,label=lk.capitalize(),zorder=3)
        wins+=(np.round(t.gdt_full-t[b],3)>0).sum(); tot+=len(t)
    ax.plot([0,0.85],[0,0.85],color=MUTED,lw=0.8,ls='--',zorder=1)
    ax.set_xlim(0,0.85); ax.set_ylim(0,0.85); ax.set_aspect('equal'); ax.grid(color='#e6e5e1',lw=0.5)
    ax.set_xlabel(f'ARI, {LBL[b]}'); ax.text(0.03,0.80,f'GDT$_m$ better: {wins}/{tot}',fontsize=8.5,color=INK)
    ax.text(0.55,0.04,'below line:\nbaseline better',fontsize=7.5,color=INK2)
axs[0].set_ylabel(r'ARI, GDT$_m$'); axs[0].legend(frameon=False,loc='lower right',bbox_to_anchor=(1.0,0.18),fontsize=8,handletextpad=0.2)
plt.tight_layout(); plt.savefig('figs/scatter_ARI.pdf',bbox_inches='tight'); plt.savefig('figs/scatter_ARI.png',dpi=200,bbox_inches='tight'); plt.close()
# ---------- 3. heatmap of gain over best baseline ----------
D=pd.DataFrame({lk:(piv('ARI',lk).gdt_full-piv('ARI',lk)[['euclid','tangent','zmp']].max(axis=1)) for lk in LK})
D=D.loc[D.mean(axis=1).sort_values(ascending=False).index]
cmap=LinearSegmentedColormap.from_list('div',['#c62f2f','#e88b84','#f0efec','#86b6ef','#1c5cab'])
v=0.12
fig,ax=plt.subplots(figsize=(4.2,6.4))
im=ax.imshow(D.values,cmap=cmap,vmin=-v,vmax=v,aspect='auto')
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        x=D.values[i,j]; ax.text(j,i,f'{x:+.3f}',ha='center',va='center',fontsize=7,color='#ffffff' if abs(x)>0.08 else INK)
ax.set_yticks(range(len(D))); ax.set_yticklabels([s.replace('_',' ') for s in D.index],fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels([l.capitalize() for l in LK]); ax.xaxis.tick_top()
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5,3,1),minor=True); ax.set_yticks(np.arange(-.5,len(D),1),minor=True)
ax.grid(which='minor',color='#fcfcfb',lw=2); ax.tick_params(which='both',length=0)
cb=fig.colorbar(im,ax=ax,fraction=0.05,pad=0.03,extend='both'); cb.set_label(r'$\Delta$ARI: GDT$_m$ $-$ best baseline',fontsize=8); cb.outline.set_visible(False)
plt.tight_layout(); plt.savefig('figs/heatmap_gain_ARI.pdf',bbox_inches='tight'); plt.savefig('figs/heatmap_gain_ARI.png',dpi=200,bbox_inches='tight'); plt.close()
# ---------- 4. runtime ----------
tm=df[df.linkage=='ward'].pivot(index='dataset',columns='weight',values='t_metric').loc[ds]
n=df.groupby('dataset').n.first().loc[ds]; tc=df.groupby(['dataset','weight']).t_cluster.sum().unstack().loc[ds,'euclid']
fig,ax=plt.subplots(figsize=(4.6,3.3))
for col,c,mk,l in [('zmp','#eda100','v','ZMP'),('tangent','#e87ba4','D','Tangent'),('gdt_full','#2a78d6','o',r'GDT$_m$')]:
    ax.scatter(n,np.maximum(tm[col],5e-3),color=c,marker=mk,s=24,edgecolor='#fcfcfb',lw=0.5,label=l+' (metric)',zorder=3)
ax.scatter(n,np.maximum(tc,5e-3),color='#52514e',marker='x',s=22,lw=1,label='3 linkages (clustering)',zorder=3)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('number of samples $n$'); ax.set_ylabel('wall-clock time (s)')
ax.grid(color='#e6e5e1',lw=0.5,which='major'); ax.legend(frameon=False,fontsize=7.5,loc='upper left')
plt.tight_layout(); plt.savefig('figs/runtime.pdf',bbox_inches='tight'); plt.savefig('figs/runtime.png',dpi=200,bbox_inches='tight'); plt.close()
print(D.round(3))

# ======================================================================
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.edgecolor':'#898781','xtick.color':'#52514e','ytick.color':'#52514e','pdf.fonttype':42})
df=pd.read_csv(INPUT); ds=list(dict.fromkeys(df.dataset)); LK=['ward','complete','average']; INK='#0b0b0b'
piv=lambda lk: df[df.linkage==lk].pivot(index='dataset',columns='weight',values='ARI').loc[ds]
D=pd.DataFrame({lk:piv(lk).gdt_full-piv(lk).euclid for lk in LK})
D=D.loc[D.mean(axis=1).sort_values(ascending=False).index]
cmap=LinearSegmentedColormap.from_list('div',['#c62f2f','#e88b84','#f0efec','#86b6ef','#1c5cab']); v=0.12
fig,ax=plt.subplots(figsize=(4.2,6.4))
im=ax.imshow(D.values,cmap=cmap,vmin=-v,vmax=v,aspect='auto')
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        x=D.values[i,j]; ax.text(j,i,f'{x:+.3f}',ha='center',va='center',fontsize=7,color='#ffffff' if abs(x)>0.08 else INK)
ax.set_yticks(range(len(D))); ax.set_yticklabels([s.replace('_',' ') for s in D.index],fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels([l.capitalize() for l in LK]); ax.xaxis.tick_top()
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5,3,1),minor=True); ax.set_yticks(np.arange(-.5,len(D),1),minor=True)
ax.grid(which='minor',color='#fcfcfb',lw=2); ax.tick_params(which='both',length=0)
cb=fig.colorbar(im,ax=ax,fraction=0.05,pad=0.03,extend='both'); cb.set_label(r'$\Delta$ARI: GDT$_m$ $-$ Euclidean',fontsize=8); cb.outline.set_visible(False)
plt.tight_layout(); plt.savefig('figs/heatmap_gain_ARI_euclid.pdf',bbox_inches='tight'); plt.savefig('figs/heatmap_gain_ARI_euclid.png',dpi=200,bbox_inches='tight')
print(D.round(3)); print('wins',(D.round(3)>0).sum().to_dict(),'losses',(D.round(3)<0).sum().to_dict())
print('mean',D.mean().round(3).to_dict(),'median',D.median().round(3).to_dict())
print('rows all positive',(D.round(3)>0).all(axis=1).sum(),'all negative',(D.round(3)<0).all(axis=1).sum())
print('>0.1 cells',(D>0.1).sum().sum(), ' < -0.1',(D<-0.1).sum().sum())

# ======================================================================
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.family':'serif','font.size':9,'xtick.color':'#52514e','ytick.color':'#52514e','pdf.fonttype':42})
df=pd.read_csv(INPUT); ds=list(dict.fromkeys(df.dataset)); LK=['ward','complete','average']; INK='#0b0b0b'
piv=lambda lk: df[df.linkage==lk].pivot(index='dataset',columns='weight',values='ARI').loc[ds]
DE=pd.DataFrame({lk:piv(lk).gdt_full-piv(lk).euclid for lk in LK})
DB=pd.DataFrame({lk:piv(lk).gdt_full-piv(lk)[['euclid','tangent','zmp']].max(axis=1) for lk in LK})
order=DE.mean(axis=1).sort_values(ascending=False).index; DE,DB=DE.loc[order],DB.loc[order]
cmap=LinearSegmentedColormap.from_list('div',['#c62f2f','#e88b84','#f0efec','#86b6ef','#1c5cab']); v=0.12
fig,axs=plt.subplots(1,2,figsize=(6.6,6.6),sharey=True,gridspec_kw=dict(wspace=0.06))
for ax,D,t in zip(axs,[DE,DB],['(a) GDT$_m$ $-$ Euclidean','(b) GDT$_m$ $-$ best baseline']):
    im=ax.imshow(D.values,cmap=cmap,vmin=-v,vmax=v,aspect='auto')
    for i in range(D.shape[0]):
        for j in range(3):
            x=D.values[i,j]; ax.text(j,i,f'{x:+.3f}',ha='center',va='center',fontsize=6.8,color='#ffffff' if abs(x)>0.08 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([l.capitalize() for l in LK],fontsize=8.5); ax.xaxis.tick_top()
    ax.set_title(t,fontsize=9.5,color=INK,pad=22)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5,3,1),minor=True); ax.set_yticks(np.arange(-.5,len(D),1),minor=True)
    ax.grid(which='minor',color='#fcfcfb',lw=2); ax.tick_params(which='both',length=0)
axs[0].set_yticks(range(len(DE))); axs[0].set_yticklabels([s.replace('_',' ') for s in DE.index],fontsize=8)
cb=fig.colorbar(im,ax=axs,fraction=0.035,pad=0.02,extend='both'); cb.set_label(r'$\Delta$ARI',fontsize=8.5); cb.outline.set_visible(False)
plt.savefig('figs/heatmap_gain_ARI_combined.pdf',bbox_inches='tight'); plt.savefig('figs/heatmap_gain_ARI_combined.png',dpi=200,bbox_inches='tight')
