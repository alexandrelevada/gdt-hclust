#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera as tabelas LaTeX do artigo: tab_summary.tex (Tabela de resumo), tab_wilcoxon.tex
(comparações pareadas) e appendix_tables.tex (conjuntos de dados + tabelas completas das 4 métricas)."""
import argparse, os, sys
_p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument('--input', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'paper', 'benchmark_long.csv'),
                help='CSV longo gerado por benchmark_hclust.py (padrão: resultados do artigo)')
_p.add_argument('--outdir', default='outputs', help='pasta de saída (padrão: outputs)')
ARGS = _p.parse_args()
INPUT = os.path.abspath(ARGS.input)
os.makedirs(ARGS.outdir, exist_ok=True); os.chdir(ARGS.outdir); os.makedirs('figs', exist_ok=True)

# ======================================================================
import pandas as pd, numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
df=pd.read_csv(INPUT); ds=list(dict.fromkeys(df.dataset))
W=['euclid','tangent','zmp','gdt','gdt_full']; LK=['ward','complete','average']; M=['ARI','AMI','FMI','V-measure']
LBL={'euclid':'Euclidean','tangent':'Tangent','zmp':'ZMP','gdt':r'GDT$_d$','gdt_full':r'GDT$_m$'}
SH={'euclid':'Euclid','tangent':'Tangent','zmp':'ZMP','gdt':r'GDT$_d$','gdt_full':r'GDT$_m$'}
def piv(m,lk): return df[df.linkage==lk].pivot(index='dataset',columns='weight',values=m).loc[ds,W]
def holm(p):
    p=np.array(p);o=np.argsort(p);k=len(p);a=np.empty(k);mx=0
    for r,i in enumerate(o): mx=max(mx,(k-r)*p[i]); a[i]=min(1,mx)
    return a
def fp(p): return f'{p:.3f}' if p>=1e-3 else r'$<$0.001'
def bold(s,b): return rf'\textbf{{{s}}}' if b else s
# ---------- Table: summary ----------
L=[r'\begin{table*}[t]',r'\centering',
r'\caption{Summary of the results on the 22 data sets: mean score and mean rank (lower is better) of each edge metric for each linkage criterion and external validity index. The best value in each column is shown in bold. The last row of each block gives the $p$-value of the Friedman test of equal mean ranks.}',
r'\label{tab:summary}',r'\begin{tabular}{@{}llcccccc@{}}',r'\toprule',
r' & & \multicolumn{2}{c}{Ward} & \multicolumn{2}{c}{Complete} & \multicolumn{2}{c}{Average} \\',
r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
r'Index & Edge metric & Mean & Rank & Mean & Rank & Mean & Rank \\',r'\midrule']
for mi,m in enumerate(M):
    stats={}
    for lk in LK:
        t=piv(m,lk); R=np.vstack([rankdata(-np.round(r,3)) for r in t.values]).mean(0)
        stats[lk]=(t.mean().values,R,friedmanchisquare(*[t[w] for w in W]).pvalue)
    for k,w in enumerate(W):
        row=[m if k==0 else '',LBL[w]]
        for lk in LK:
            mu,R,_=stats[lk]
            row+= [bold(f'{mu[k]:.3f}',np.isclose(np.round(mu[k],3),np.round(mu.max(),3))), bold(f'{R[k]:.2f}',np.isclose(np.round(R[k],2),np.round(R.min(),2)))]
        L.append(' & '.join(row)+r' \\')
    L.append(r' & \textit{Friedman $p$} & \multicolumn{2}{c}{'+fp(stats['ward'][2])+r'} & \multicolumn{2}{c}{'+fp(stats['complete'][2])+r'} & \multicolumn{2}{c}{'+fp(stats['average'][2])+r'} \\')
    if mi<3: L.append(r'\midrule')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
open('tab_summary.tex','w').write('\n'.join(L)+'\n')
# ---------- Table: pairwise Wilcoxon ----------
L=[r'\begin{table*}[t]',r'\centering',
r'\caption{Pairwise comparison of GDT$_m$ with each competing edge metric over the 22 data sets (ARI and AMI). W/T/L: number of data sets in which GDT$_m$ wins, ties (equal to three decimals) or loses; $\tilde\Delta$: median paired difference; $p$: two-sided Wilcoxon signed-rank test with Holm correction over the comparisons in each row block. Differences significant at $\alpha=0.05$ are shown in bold. The last row of each block (GDT$_m$ vs.\ GDT$_d$) is the rank ablation and is not included in the Holm correction.}',
r'\label{tab:wilcoxon}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{@{}llccccccccc@{}}',r'\toprule',
r' & & \multicolumn{3}{c}{Ward} & \multicolumn{3}{c}{Complete} & \multicolumn{3}{c}{Average} \\',
r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}',
r'Index & GDT$_m$ vs. & W/T/L & $\tilde\Delta$ & $p$ & W/T/L & $\tilde\Delta$ & $p$ & W/T/L & $\tilde\Delta$ & $p$ \\',r'\midrule']
for mi,m in enumerate(['ARI','AMI']):
    cells={}
    for lk in LK:
        t=piv(m,lk); ps=[]; rows=[]
        for b in ['euclid','tangent','zmp','gdt']:
            d=t.gdt_full-t[b]; dr=np.round(d,3)
            ps.append(wilcoxon(t.gdt_full,t[b]).pvalue); rows.append(((dr>0).sum(),(dr==0).sum(),(dr<0).sum(),np.median(d)))
        adj=list(holm(ps[:3]))+[ps[3]]
        cells[lk]=[(r,a) for r,a in zip(rows,adj)]
    for k,b in enumerate(['euclid','tangent','zmp','gdt']):
        row=[m if k==0 else '',SH[b]]
        for lk in LK:
            (w,t_,l,md),a=cells[lk][k]
            row+=[f'{w}/{t_}/{l}',f'{md:+.3f}',bold(fp(a),a<0.05)]
        if k==3: L.append(r'\cdashline{2-11}' if False else r'\cmidrule(l){2-11}')
        L.append(' & '.join(row)+r' \\')
    if mi==0: L.append(r'\midrule')
L+=[r'\bottomrule',r'\end{tabular}}',r'\end{table*}']
open('tab_wilcoxon.tex','w').write('\n'.join(L)+'\n')
# ---------- Appendix: datasets + full tables ----------
meta=df.groupby('dataset').first().loc[ds]
A=[r'\begin{table}[t]',r'\centering',
r'\caption{Data sets used in the experiments: number of samples $n$ after deduplication and subsampling, number of attributes $m$ before and after preprocessing (one-hot encoding and PCA to 50 components when $m>100$), estimated intrinsic dimension $\hat d$ (Levina--Bickel, $k=15$; the value used, capped at 10, in parentheses) and number of classes $c$.}',
r'\label{tab:datasets}',r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',r'Data set & $n$ & $m$ & $m$ used & $\hat d$ & $c$ \\',r'\midrule']
for d_ in ds:
    r=meta.loc[d_]; A.append(f"{d_.replace('_',' ')} & {r.n} & {r.m_original} & {r.m_used} & {r.d_est:.2f} ({r.d}) & {r.k_clusters} \\\\")
A+=[r'\bottomrule',r'\end{tabular}',r'\end{table}','']
NAME={'ARI':'Adjusted Rand Index (ARI)','AMI':'Adjusted mutual information (AMI)','FMI':'Fowlkes--Mallows index (FMI)','V-measure':'V-measure'}
for m in M:
    A+=[r'\begin{table*}[p]',r'\centering',
    rf'\caption{{{NAME[m]} of geodesic hierarchical clustering on each data set for every edge metric and linkage criterion. The best edge metric for each data set and linkage is shown in bold.}}',
    rf"\label{{tab:full-{m.lower().replace('-','')}}}",r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{@{}l'+'c'*15+'@{}}',r'\toprule',
    r' & \multicolumn{5}{c}{Ward} & \multicolumn{5}{c}{Complete} & \multicolumn{5}{c}{Average} \\',
    r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(lr){12-16}',
    'Data set & '+' & '.join(SH[w] for _ in LK for w in W)+r' \\',r'\midrule']
    T={lk:piv(m,lk) for lk in LK}
    for d_ in ds:
        cells=[]
        for lk in LK:
            v=np.round(T[lk].loc[d_].values,3)
            cells+=[bold(f'{x:.3f}',x==v.max()) for x in v]
        A.append(d_.replace('_',r'\_')+' & '+' & '.join(cells)+r' \\')
    A.append(r'\midrule')
    for lab,fn in [('Mean',lambda t:t.mean().values),('Mean rank',lambda t:np.vstack([rankdata(-np.round(r,3)) for r in t.values]).mean(0)),
                   ('Wins',lambda t:np.array([(np.round(t[w],3)==np.round(t.max(axis=1),3)).sum() for w in W]))]:
        cells=[]
        for lk in LK:
            v=fn(T[lk]); best=v.min() if lab=='Mean rank' else v.max()
            fmt=(lambda x:f'{x:.2f}') if lab=='Mean rank' else ((lambda x:f'{int(x)}') if lab=='Wins' else (lambda x:f'{x:.3f}'))
            cells+=[bold(fmt(x),np.isclose(x,best)) for x in v]
        A.append(lab+' & '+' & '.join(cells)+r' \\')
    A+=[r'\bottomrule',r'\end{tabular}}',r'\end{table*}','']
open('appendix_tables.tex','w').write('\n'.join(A))
print('tabelas escritas em', os.getcwd())
