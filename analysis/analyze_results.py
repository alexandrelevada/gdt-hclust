#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testes estatísticos do artigo: Friedman, Wilcoxon-Holm (GDT_m e GDT_d vs. baselines), V/E/D,
correlações dos ganhos com as características dos conjuntos, ablação de posto e tempos.
Imprime tudo na tela; use  > outputs/analysis.txt  para salvar."""
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
df = pd.read_csv(INPUT)
W = ['euclid','tangent','zmp','gdt','gdt_full']; LK=['ward','complete','average']; M=['ARI','AMI','FMI','V-measure']
ds = list(dict.fromkeys(df.dataset))
print(len(ds), 'datasets')
def table(metric, lk):
    t = df[df.linkage==lk].pivot(index='dataset', columns='weight', values=metric).loc[ds, W]
    return t
def holm(p):
    p=np.array(p); o=np.argsort(p); m=len(p); adj=np.empty(m); mx=0
    for r,i in enumerate(o):
        mx=max(mx,(m-r)*p[i]); adj[i]=min(1,mx)
    return adj
for metric in M:
    print('=====', metric)
    for lk in LK+['pooled']:
        t = pd.concat([table(metric,l) for l in LK]) if lk=='pooled' else table(metric,lk)
        R = np.vstack([rankdata(-np.round(r,3)) for r in t.values])
        fr = friedmanchisquare(*[t[w] for w in W])
        line=f'{lk:9s} Friedman p={fr.pvalue:.2e} | ranks '+' '.join(f'{w}:{R[:,k].mean():.2f}' for k,w in enumerate(W))
        print(line)
        for prop in ['gdt_full','gdt']:
            ps=[];out=[]
            for b in ['euclid','tangent','zmp']:
                d = np.round(t[prop]-t[b],3)
                wtl=( (d>0).sum(), (d==0).sum(), (d<0).sum())
                p = wilcoxon(t[prop], t[b], zero_method='wilcox', alternative='two-sided').pvalue
                ps.append(p); out.append((b,wtl,np.median(t[prop]-t[b]),(t[prop]-t[b]).mean()))
            adj=holm(ps)
            print('   ',prop,' | '.join(f'vs {b}: W/T/L={w[0]}/{w[1]}/{w[2]} medΔ={md:+.3f} meanΔ={mn:+.3f} p={p:.3g} pHolm={a:.3g}' for (b,w,md,mn),p,a in zip(out,ps,adj)))

# ======================================================================
import pandas as pd, numpy as np
from scipy.stats import spearmanr, wilcoxon
df = pd.read_csv(INPUT); LK=['ward','complete','average']
meta = df.groupby('dataset').first()[['n','m_original','m_used','d','d_est','k_clusters']]
ds = list(dict.fromkeys(df.dataset)); meta=meta.loc[ds]
meta['codim']=meta.m_used-meta.d
P = {lk: df[df.linkage==lk].pivot(index='dataset',columns='weight',values='ARI').loc[ds] for lk in LK}
best_base = {lk: P[lk][['euclid','tangent','zmp']].max(axis=1) for lk in LK}
# gain over best baseline, and vs zmp
for lk in LK:
    g = P[lk].gdt_full - best_base[lk]; gz = P[lk].gdt_full - P[lk].zmp
    print(lk, 'beats best baseline in', (g>0).sum(), 'of 22; median gain', round(g.median(),3))
G = pd.DataFrame({lk: P[lk].gdt_full - P[lk].zmp for lk in LK}); G['mean']=G.mean(axis=1)
Ge = pd.DataFrame({lk: P[lk].gdt_full - P[lk].euclid for lk in LK}); Ge['mean']=Ge.mean(axis=1)
for col in ['n','m_used','d','codim','k_clusters']:
    x = np.log(meta[col]) if col=='n' else meta[col]
    r1=spearmanr(meta[col],G['mean']); r2=spearmanr(meta[col],Ge['mean'])
    print(f'{col:10s} rho(Δ vs zmp)={r1.correlation:+.2f} p={r1.pvalue:.3f} | rho(Δ vs euclid)={r2.correlation:+.2f} p={r2.pvalue:.3f}')
print(pd.concat([meta[['n','m_used','d','codim']],G.round(3),Ge['mean'].round(3).rename('vsEuc')],axis=1).sort_values('mean'))
# rank ablation gdt_full vs gdt
for m in ['ARI','AMI']:
    for lk in LK:
        t=df[df.linkage==lk].pivot(index='dataset',columns='weight',values=m)
        d=np.round(t.gdt_full-t.gdt,3)
        print(m,lk,'GDT_m vs GDT W/T/L',(d>0).sum(),(d==0).sum(),(d<0).sum(),'p=%.3g'%wilcoxon(t.gdt_full,t.gdt).pvalue, 'medΔ=%+.3f'%np.median(t.gdt_full-t.gdt))
# timing
tm = df[df.linkage=='ward'].pivot(index='dataset',columns='weight',values='t_metric').loc[ds]
tc = df.groupby(['dataset','weight']).t_cluster.sum().unstack().loc[ds]
print(pd.concat([meta.n, tm[['zmp','tangent','gdt','gdt_full']], tc['euclid'].rename('t_clust3')],axis=1).sort_values('n'))
