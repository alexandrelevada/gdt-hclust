#!/usr/bin/env python3
"""Graphical abstract: duas faixas senoidais, arestas euclidianas vs. GDT_m (complete linkage)."""
import sys; import os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch
import hierarchical_metric_clustering as H, metric_tensor_spring as MT
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score as ari
plt.rcParams.update({'font.family':'serif','pdf.fonttype':42,'mathtext.fontset':'dejavuserif'})
INK='#0b0b0b'; INK2='#52514e'; MUT='#898781'; BG='#fcfcfb'; C1='#2a78d6'; C2='#eb6834'; RED='#c62f2f'
rng=np.random.default_rng(3); n=260
X=[];y=[]
for c,off in enumerate([0,0.4]):
    t=np.sort(rng.uniform(0,6,n)); X.append(np.c_[t,0.45*np.sin(0.9*t)+off+rng.normal(0,0.04,n)]); y+=[c]*n
X=np.vstack(X); y=np.array(y)
A=MT.knn_graph(X,10,'distance'); nc,_=connected_components(A)
if nc>1: A,_=MT.mst_bridges(X,A)
A=A.tocoo(); i,j=A.row,A.col
U,c=H.gdt_lowrank(X,2,10,15,10,1.0,1e-2,verbose=False)
Le=A.data; Lg=H.edge_lengths(X,i,j,U,c)
def clus(l):
    lab=H.cluster(H.geodesics(A,l),'complete',2,10)
    if (lab==y).mean()<0.5: lab=1-lab
    return lab, ari(y,lab)
labE,aE=clus(Le); labG,aG=clus(Lg); _,aZ=clus(H.zmp_lengths(X,i,j,7))
print('ARI euclid %.2f zmp %.2f gdt %.2f'%(aE,aZ,aG))
keep=i<j; I,J=i[keep],j[keep]; Le,Lg=Le[keep],Lg[keep]; cross=y[I]!=y[J]
fig=plt.figure(figsize=(13,6.2),facecolor='white')
# title
fig.text(0.5,0.965,'Not All Edges Are Equal',ha='center',fontsize=19,color=INK,weight='bold')
fig.text(0.5,0.915,'Geodesic Diffusion Tensors as anisotropic metrics for hierarchical clustering',ha='center',fontsize=12,color=INK2,style='italic')
W=0.295; H_=0.56; Y=0.30; xs=[0.035,0.3525,0.67]
axs=[fig.add_axes([x,Y,W,H_]) for x in xs]
def base(ax,title,sub):
    ax.set_xlim(-0.25,6.25); ax.set_ylim(-1.45,1.95); ax.set_aspect('auto'); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0,0),1,1,transform=ax.transAxes,boxstyle='round,pad=0.0,rounding_size=0.03',fc='#f6f5f2',ec='#e6e5e1',lw=1,zorder=0,clip_on=False))
    ax.text(0.04,0.95,title,transform=ax.transAxes,fontsize=12,weight='bold',color=INK,va='top')
    ax.text(0.04,0.875,sub,transform=ax.transAxes,fontsize=9,color=INK2,va='top')
def edges(ax,hl_color):
    seg=np.stack([X[I],X[J]],1)
    ax.add_collection(LineCollection(seg[~cross],colors='#bdbbb6',linewidths=0.6,zorder=1))
    ax.add_collection(LineCollection(seg[cross],colors=hl_color,linewidths=1.4,zorder=4))
def pts(ax,lab,s=9):
    ax.scatter(X[:,0],X[:,1],c=np.where(lab==0,C1,C2),s=s,edgecolor=BG,linewidth=0.3,zorder=3)
# ---- panel 1
ax=axs[0]; base(ax,'1  Euclidean edges','every edge measured with the same ruler')
edges(ax,INK); pts(ax,labE)
ax.text(0.96,0.135,f'black: the only {cross.sum()} edges joining the strips are\njust {np.median(Le[cross])/np.median(Le[~cross]):.1f}$\\times$ longer than edges within them',transform=ax.transAxes,ha='right',fontsize=8.3,color=INK)
ax.text(0.04,0.04,f'complete linkage: ARI = {aE:.2f}',transform=ax.transAxes,fontsize=9.5,color=INK,weight='bold')
# ---- panel 2
ax=axs[1]; base(ax,'2  Geodesic Diffusion Tensor','local metric $G_i$ estimated at every sample')
ax.scatter(X[:,0],X[:,1],c='#a9a7a1',s=5,zorder=2,lw=0)
sel=np.r_[np.arange(6,n,16),np.arange(n+14,2*n,16)]
for a in sel:
    u=U[a].astype(float); cc=c[a]; ax_=1/np.sqrt(cc)        # unit ball of G: semi-axes 1/sqrt(c)
    k=np.argmax(ax_); ang=np.degrees(np.arctan2(u[1,k],u[0,k]))
    s=0.085
    ax.add_patch(Ellipse(X[a],2*s*ax_[k],2*s*ax_[1-k],angle=ang,fc=(C1 if y[a]==0 else C2),alpha=0.35,ec=(C1 if y[a]==0 else C2),lw=1.0,zorder=3))
ax.text(0.5,0.135,r'$h_i=\frac{2}{\varepsilon}\sum_j P_{ij}\,\Delta x_{ij}\Delta x_{ij}^{\top},\qquad G_i=(\bar h_i)^{+}$',transform=ax.transAxes,ha='center',fontsize=10.5,color=INK)
ax.text(0.5,0.045,'unit balls of $G_i$: long along the data, thin across it',transform=ax.transAxes,ha='center',fontsize=8.5,color=INK2)
# ---- panel 3
ax=axs[2]; base(ax,'3  Anisotropic edge lengths','$\\ell_{ij}=\\sqrt{\\Delta x_{ij}^{\\top}\\,\\frac{1}{2}(G_i+G_j)\\,\\Delta x_{ij}}$')
edges(ax,RED); pts(ax,labG)
ax.text(0.04,0.04,f'complete linkage: ARI = {aG:.2f}',transform=ax.transAxes,fontsize=9.5,color=INK,weight='bold')
ax.text(0.04,0.105,f'self-tuning ZMP: ARI = {aZ:.2f}',transform=ax.transAxes,fontsize=8.5,color=INK2)
ax.text(0.96,0.20,f'red: the same edges become {np.median(Lg[cross])/np.median(Lg[~cross]):.1f}$\\times$ longer\nunder the local metrics',transform=ax.transAxes,ha='right',fontsize=8.3,color=RED)
# arrows between panels
for x0 in [xs[0]+W, xs[1]+W]:
    fig.patches.append(FancyArrowPatch((x0+0.003,Y+H_/2),(x0+0.0195,Y+H_/2),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=16,color=INK,lw=1.4))
# ---- pipeline strip
steps=['$k$-NN graph','geodesic, density-\nnormalized kernel','self-loop-free\nsecond moment $\\to G_i$','metric edge lengths\n$\\to$ graph geodesics','any linkage\n(Ward, complete, average)']
bx0,bw,by,bh,gapx=0.035,0.158,0.085,0.115,0.0315
for k,s_ in enumerate(steps):
    x=bx0+k*(bw+gapx)
    hl = k in (1,2)
    fig.patches.append(FancyBboxPatch((x,by),bw,bh,transform=fig.transFigure,boxstyle='round,pad=0.004,rounding_size=0.012',
                       fc=('#e8f1fc' if hl else 'white'),ec=(C1 if hl else MUT),lw=1.1))
    fig.text(x+bw/2,by+bh/2,s_,ha='center',va='center',fontsize=9.3,color=INK)
    if k<4: fig.patches.append(FancyArrowPatch((x+bw+0.004,by+bh/2),(x+bw+gapx-0.004,by+bh/2),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=12,color=MUT,lw=1.1))
fig.text(bx0+bw+gapx+bw+gapx/2,by-0.035,'GDT (unsupervised, $O(nmr)$ memory)',ha='center',fontsize=8.5,color=C1)
fig.text(0.5,0.018,'22 real data sets, 4 validity indices: best mean rank under Ward, complete and average linkage; gains come from directions normal to the data',
         ha='center',fontsize=9.5,color=INK2)
os.makedirs('outputs',exist_ok=True); fig.savefig('outputs/graphical_abstract.pdf'); fig.savefig('outputs/graphical_abstract.png',dpi=220)
