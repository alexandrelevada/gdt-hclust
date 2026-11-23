#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hierarchical_metric_clustering.py
=================================

Clustering hierárquico (Ward, single, complete, average) sobre distâncias geodésicas no
grafo k-NN do ESPAÇO ORIGINAL, com pesos de aresta corrigidos por métricas locais:

  euclid    ‖Δx‖                                       (geodésica usual, tipo Isomap)
  tangent   ‖P_T Δx‖                                    projeção no tangente local (PCA, posto d)
  zmp       ‖Δx‖ / √(σ_i σ_j)                           escala local de Zelnik-Manor & Perona (2004),
                                                        σ_i = distância ao K-ésimo vizinho (--k-zmp, padrão 7)
  sigma    √(Δxᵀ Σ⁺ Δx)                                Σ = covariância dos k vizinhos (posto d) — métrica do MeCuCo
  gdt       √(Δxᵀ ½(G_i + G_j) Δx)                      G = tensor GDT-h com φ = x, posto d
  gdt_full  idem, posto cheio (só quando m ≤ --full-max-dim)

As métricas locais são guardadas em forma fatorada (m × d por ponto), então a memória
é O(n·m·d) — viável para m grande (ex.: MNIST, m = 784).

Uso típico
----------
    python hierarchical_metric_clustering.py --dataset digits
    python hierarchical_metric_clustering.py --dataset wine --linkages ward complete average
    python hierarchical_metric_clustering.py --source openml --dataset mnist_784 --max-samples 3000 --max-support 200
    python hierarchical_metric_clustering.py --source openml --data-id 61 --weights euclid gdt
    python hierarchical_metric_clustering.py --dataset blobs --n-samples 1500 --n-clusters 3

Métricas de avaliação (quando há rótulos)
-----------------------------------------
    ARI         Adjusted Rand Index
    MI          informação mútua — por padrão a AJUSTADA (AMI, corrigida para o acaso);
                --mi-variant nmi (normalizada) ou mi (bruta, em nats, sem limite superior)
    FMI         índice de Fowlkes–Mallows
    V-measure   média harmônica de homogeneidade e completude

Saídas (prefixo --out, padrão hclust_<fonte>_<dataset>)
-------------------------------------------------------
    <prefixo>_results.csv     formato longo: todas as métricas e tamanhos dos grupos, por peso × linkage
    <prefixo>_<METRICA>.csv   uma tabela por métrica (linhas = linkage, colunas = peso)
    <prefixo>_tables.tex      as quatro tabelas em LaTeX (booktabs), melhor peso em negrito
    <prefixo>_metrics.png     gráfico de barras com as quatro métricas
    <prefixo>_labels.npz      rótulos atribuídos (com --save-labels)

Leitura dos resultados (dos nossos experimentos)
------------------------------------------------
- Ward e single linkage praticamente não mudam com a correção.
- Complete linkage tende a melhorar com métricas que reescalam localmente (sigma, gdt):
  o ganho vem da normalização de escala entre regiões densas e esparsas, não de
  geodésicas mais fiéis.
"""
import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
                             mutual_info_score, fowlkes_mallows_score, v_measure_score)
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metric_tensor_spring as MT                                # noqa: E402  (biblioteca central)

warnings.filterwarnings('ignore')
WEIGHTS = ['euclid', 'tangent', 'zmp', 'sigma', 'gdt', 'gdt_full']
LINKAGES = ['ward', 'single', 'complete', 'average']
LABEL = {'euclid': 'euclidiano', 'tangent': 'tangente', 'zmp': 'ZMP (self-tuning)', 'sigma': 'Σ⁻¹ local',
         'gdt': 'GDT posto d', 'gdt_full': 'GDT posto m'}
LABEL_TEX = {'euclid': 'Euclid', 'tangent': 'Tangente', 'zmp': 'ZMP', 'sigma': r'$\Sigma^{-1}$',
             'gdt': 'GDT', 'gdt_full': r'GDT$_m$'}
MI_VARIANTS = {'ami': ('AMI', adjusted_mutual_info_score, 'informação mútua ajustada (AMI)'),
               'nmi': ('NMI', normalized_mutual_info_score, 'informação mútua normalizada (NMI)'),
               'mi':  ('MI', mutual_info_score, 'informação mútua (nats)')}


def metric_set(mi_variant='ami'):
    """Lista (nome, função, descrição) das quatro métricas de avaliação."""
    mi_name, mi_fn, mi_desc = MI_VARIANTS[mi_variant]
    return [('ARI', adjusted_rand_score, 'Adjusted Rand Index (ARI)'),
            (mi_name, mi_fn, mi_desc),
            ('FMI', fowlkes_mallows_score, 'índice de Fowlkes–Mallows (FMI)'),
            ('V-measure', v_measure_score, 'V-measure')]


def write_metric_tables(prefix, rows, weights, linkages, metrics, title):
    """Uma tabela por métrica (CSV, texto e LaTeX): linhas = linkage, colunas = peso."""
    def val(w, lk, mt):
        return next(r[mt] for r in rows if r['peso'] == w and r['linkage'] == lk)
    txt, tex = [], ['% requer \\usepackage{booktabs}']
    for mt, _, desc in metrics:
        fn = f"{prefix}_{mt.replace('-', '')}.csv"
        with open(fn, 'w', newline='') as fh:
            wr = csv.writer(fh); wr.writerow(['linkage'] + weights)
            for lk in linkages:
                wr.writerow([lk] + [f'{val(w, lk, mt):.4f}' for w in weights])
        fmt = '{:.3f}' if mt != 'MI' else '{:.3f}'
        txt.append(f'\n  {mt} — linhas: linkage, colunas: peso das arestas')
        txt.append(f'  {"":10s}' + ''.join(f'{LABEL[w]:>14s}' for w in weights))
        for lk in linkages:
            v = [val(w, lk, mt) for w in weights]
            best = round(max(v), 3)
            txt.append(f'  {lk:10s}' + ''.join(
                (f'{x:13.3f}*' if round(x, 3) == best else f'{x:14.3f}') for x in v))
        tex += [r'\begin{table}[t]', r'\centering',
                rf'\caption{{{desc} do clustering hierárquico sobre geodésicas no grafo $k$-NN '
                rf'do espaço original --- {title}. Em negrito, o melhor peso de aresta para cada ligação.}}',
                rf"\label{{tab:{prefix.split('/')[-1]}-{mt.lower().replace('-', '')}}}",
                r'\begin{tabular}{l' + 'c' * len(weights) + '}', r'\toprule',
                'Ligação & ' + ' & '.join(LABEL_TEX[w] for w in weights) + r' \\', r'\midrule']
        for lk in linkages:
            v = [val(w, lk, mt) for w in weights]
            best = round(max(v), 3)
            cells = [(rf'\textbf{{{fmt.format(x)}}}' if round(x, 3) == best else fmt.format(x)) for x in v]
            tex.append(f'{lk.capitalize()} & ' + ' & '.join(cells) + r' \\')
        tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    with open(f'{prefix}_tables.tex', 'w') as fh:
        fh.write('\n'.join(tex))
    return '\n'.join(txt)


# ══════════════════════════════════════════════════════════════════════════════
# Métricas locais em forma fatorada:  M_i = U_i diag(c_i) U_iᵀ,  U_i ∈ ℝ^{m×r}
# ══════════════════════════════════════════════════════════════════════════════
def top_factor(W, r):
    """Top-r autopares de WᵀW (W: s×m), pelo caminho mais barato. Retorna (U m×r, μ r)."""
    s, m = W.shape
    if s <= m:
        mu, v = np.linalg.eigh(W @ W.T)
        mu, v = mu[::-1][:r], v[:, ::-1][:, :r]
        keep = mu > 1e-12 * max(mu.max(), 1e-300)
        mu, v = mu[keep], v[:, keep]
        U = (W.T @ v) / np.sqrt(mu)
    else:
        mu, U = np.linalg.eigh(W.T @ W)
        mu, U = mu[::-1][:r], U[:, ::-1][:, :r]
    return U, np.clip(mu, 0, None)


def pad(U, c, m, r):
    """Completa com zeros até r colunas (pontos com posto local menor)."""
    if U.shape[1] < r:
        U = np.hstack([U, np.zeros((m, r - U.shape[1]))]); c = np.r_[c, np.zeros(r - len(c))]
    return U, c


def local_pca_ops(X, d, k):
    """Projetor tangente (c = 1) e Σ⁺ de posto d (c = 1/λ) a partir dos k vizinhos."""
    n, m = X.shape
    nb = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    U = np.zeros((n, m, d), np.float32); lam = np.zeros((n, d))
    for i in range(n):
        Z = X[nb[i]] - X[nb[i]].mean(0)
        Ui, mu = top_factor(Z, d)
        Ui, mu = pad(Ui, mu / (k - 1), m, d)
        U[i], lam[i] = Ui, mu
    floor = 1e-6 * np.median(lam[lam > 0])
    return U, np.ones_like(lam), 1.0 / np.maximum(lam, floor)


def gdt_lowrank(X, d, k_graph=10, k_eps=15, k_smooth=10, eps_scale=1.0, h_floor=1e-2,
                block=512, max_support=None, verbose=True):
    """GDT-h com φ = x em forma fatorada: retorna U (n×m×d) e c = 1/λ (n×d), G_i = U diag(c) Uᵀ.
    Mesmo kernel do metric_tensor_spring.gdt_h (peso pela corda, suporte geodésico ≤ 3√ε,
    α = 1, sem o próprio ponto); a suavização de h no grafo é feita sobre os fatores de posto 2d."""
    n, m = X.shape
    r = min(2 * d, m)
    r_k = NearestNeighbors(n_neighbors=k_eps).fit(X).kneighbors(return_distance=True)[0][:, -1]
    eps = eps_scale * np.median(r_k) ** 2
    A = MT.knn_graph(X, k_graph, 'distance')
    Sup = MT.geodesic_support(A, 3.0 * np.sqrt(eps), block)
    Sup = (Sup + (A > 0)).tocsr(); Sup.setdiag(False); Sup.eliminate_zeros()
    i, j = Sup.nonzero()
    chord2 = MT.pair_sqdist(X, i, j, block)
    if max_support:                                       # mantém os pares mais próximos por ponto
        keep = np.zeros(len(i), bool)
        indptr = np.concatenate([[0], np.cumsum(np.bincount(i, minlength=n))])
        for a_ in range(n):
            lo, hi = indptr[a_], indptr[a_ + 1]
            keep[lo:hi] = True if hi - lo <= max_support else False
            if hi - lo > max_support:
                keep[lo + np.argpartition(chord2[lo:hi], max_support)[:max_support]] = True
        i, j, chord2 = i[keep], j[keep], chord2[keep]
    K = sparse.csr_matrix((np.exp(-chord2 / eps), (i, j)), shape=(n, n))
    K = K.maximum(K.T)
    q = np.asarray(K.sum(1)).ravel() + 1.0
    P = MT.row_normalize(sparse.diags(1.0 / q) @ K @ sparse.diags(1.0 / q)).tocsr()
    # fatores de h_i = (2/ε) Σ_j P_ij Δx Δxᵀ  ≈  F_i F_iᵀ, F_i ∈ ℝ^{m×r}
    F = np.zeros((n, m, r), np.float32)
    for a in range(n):
        cols = P.indices[P.indptr[a]:P.indptr[a + 1]]; w = P.data[P.indptr[a]:P.indptr[a + 1]]
        W = np.sqrt(2.0 / eps * w)[:, None] * (X[cols] - X[a])
        Ua, mu = top_factor(W, r); Ua, mu = pad(Ua, mu, m, r)
        F[a] = Ua * np.sqrt(mu)
    # suavização: h̄_i = Σ_j S_ij h_j  →  top-d de [√S_ij F_j]
    S = MT.row_normalize(MT.knn_graph(X, k_smooth, 'connectivity') + sparse.identity(n)).tocsr()
    U = np.zeros((n, m, d), np.float32); lam = np.zeros((n, d))
    for a in range(n):
        cols = S.indices[S.indptr[a]:S.indptr[a + 1]]; w = S.data[S.indptr[a]:S.indptr[a + 1]]
        B = np.concatenate([np.sqrt(wj) * F[jj].T for jj, wj in zip(cols, w)], 0)   # (Σr, m)
        Ua, mu = top_factor(B.astype(np.float64), d); Ua, mu = pad(Ua, mu, m, d)
        U[a], lam[a] = Ua, mu
    floor = h_floor * np.median(lam[lam > 0])
    n_floor = int((lam < floor).any(1).sum())
    if verbose:
        print(f'  GDT: ε = {eps:.4g}, pares no suporte (mediana) = {np.median(np.diff(P.indptr)):.0f}, '
              f'{n_floor} ponto(s) regularizado(s)')
    return U, 1.0 / np.maximum(lam, floor)


def zmp_scales(X, k_zmp=7):
    """σ_i de Zelnik-Manor & Perona (2004): distância de x_i ao seu K-ésimo vizinho (sem contar x_i)."""
    k_zmp = int(min(max(k_zmp, 1), len(X) - 1))
    dist = NearestNeighbors(n_neighbors=k_zmp).fit(X).kneighbors(return_distance=True)[0]
    sig = dist[:, -1]
    return np.maximum(sig, 1e-6 * max(np.median(sig), 1e-300))


def zmp_lengths(X, i, j, k_zmp=7, block=512):
    """Comprimento de aresta do self-tuning (ZMP): ℓ_ij = ‖x_i − x_j‖ / √(σ_i σ_j).
    É o comprimento induzido pela afinidade W_ij = exp(−‖x_i − x_j‖² / (σ_i σ_j)) = exp(−ℓ_ij²),
    ou seja, a métrica local isotrópica e conforme  M_i = I / σ_i² (média geométrica nos extremos)."""
    sig = zmp_scales(X, k_zmp)
    d = np.sqrt(MT.pair_sqdist(X, np.asarray(i), np.asarray(j), block))
    return d / np.sqrt(sig[i] * sig[j])


def edge_lengths(X, i, j, U, c, chunk=20000):
    """ℓ_ij = √(Δxᵀ ½(M_i + M_j) Δx), M = U diag(c) Uᵀ, em blocos de arestas."""
    out = np.empty(len(i))
    for s in range(0, len(i), chunk):
        a, b = i[s:s + chunk], j[s:s + chunk]
        dX = (X[b] - X[a]).astype(np.float32)
        qa = (np.einsum('em,emr->er', dX, U[a]) ** 2 * c[a]).sum(1)
        qb = (np.einsum('em,emr->er', dX, U[b]) ** 2 * c[b]).sum(1)
        out[s:s + chunk] = np.sqrt(np.clip(0.5 * (qa + qb), 1e-24, None))
    return out


def geodesics(A, lengths):
    B = A.copy(); B.data = np.maximum(lengths, 1e-12)
    D = shortest_path(B.tocsr(), directed=False)
    if not np.isfinite(D).all():
        D[~np.isfinite(D)] = 3 * D[np.isfinite(D)].max()
    return D


# ══════════════════════════════════════════════════════════════════════════════
# Clustering
# ══════════════════════════════════════════════════════════════════════════════
def cluster(D, method, c, mds_dims, seed=0):
    if method == 'ward':                                  # Ward exato sobre o MDS das geodésicas
        Z = MT.classical_mds(D, mds_dims)
        return AgglomerativeClustering(n_clusters=c, linkage='ward').fit_predict(Z)
    return fcluster(linkage(squareform(D, checks=False), method=method), c, criterion='maxclust') - 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group('dados')
    g.add_argument('--source', choices=['sklearn', 'openml'], default='sklearn')
    g.add_argument('--dataset', default='digits')
    g.add_argument('--data-id', type=int, default=None)
    g.add_argument('--version', default=None)
    g.add_argument('--n-samples', type=int, default=2000, help='para geradores make_*')
    g.add_argument('--max-samples', type=int, default=3000, help='subamostragem (0 = todas)')
    g.add_argument('--pca', type=float, default=None, help='int = componentes; fração = variância')
    g.add_argument('--no-standardize', action='store_true')
    g = p.add_argument_group('geometria')
    g.add_argument('--dim', type=int, default=None, help='dimensão intrínseca d (padrão: estimada)')
    g.add_argument('--dim-estimator', choices=['levina-bickel', 'twonn'], default='levina-bickel')
    g.add_argument('--max-dim', type=int, default=10)
    g.add_argument('--k', type=int, default=15, help='k do grafo das geodésicas e da PCA local')
    g.add_argument('--k-graph', type=int, default=10, help='k do grafo do suporte geodésico do GDT')
    g.add_argument('--k-eps', type=int, default=15)
    g.add_argument('--k-smooth', type=int, default=10)
    g.add_argument('--eps-scale', type=float, default=1.0)
    g.add_argument('--h-floor', type=float, default=1e-2)
    g.add_argument('--max-support', type=int, default=None,
                   help='máximo de pares no kernel do GDT por ponto (ex.: 200; acelera muito para m grande)')
    g.add_argument('--full-max-dim', type=int, default=150, help='m máximo para gdt_full')
    g.add_argument('--k-zmp', type=int, default=7,
                   help='K do self-tuning (σ_i = distância ao K-ésimo vizinho; ZMP usam K = 7)')
    g = p.add_argument_group('clustering')
    g.add_argument('--weights', nargs='+', default=['euclid', 'tangent', 'zmp', 'sigma', 'gdt', 'gdt_full'],
                   choices=WEIGHTS)
    g.add_argument('--linkages', nargs='+', default=['ward', 'single', 'complete'], choices=LINKAGES)
    g.add_argument('--n-clusters', type=int, default=None, help='padrão: nº de classes do dataset')
    g.add_argument('--mds-dims', type=int, default=None, help='dimensões do MDS para Ward (padrão max(10, d))')
    g.add_argument('--mi-variant', choices=['ami', 'nmi', 'mi'], default='ami',
                   help='informação mútua: ajustada (padrão), normalizada ou bruta')
    g = p.add_argument_group('saída')
    g.add_argument('--out', default=None)
    g.add_argument('--save-labels', action='store_true')
    g.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    t0 = time.time()
    name = a.dataset if a.data_id is None else f'openml_{a.data_id}'
    print(f'[1/4] carregando {a.source}:{name}')
    X, y = (MT.load_sklearn(a.dataset, a.n_samples, a.seed) if a.source == 'sklearn'
            else MT.load_openml(a.dataset, a.data_id, a.version))
    X, y = MT.prepare(X, y, a.max_samples, a.seed, standardize=not a.no_standardize, pca=a.pca)
    n, m = X.shape
    has_y = y is not None and len(np.unique(y)) > 1 and len(np.unique(y)) < n / 2
    c = a.n_clusters or (len(np.unique(y)) if has_y else None)
    if c is None:
        sys.exit('Sem rótulos discretos: informe --n-clusters.')
    if has_y:
        y = np.unique(np.asarray(y).astype(str), return_inverse=True)[1]
    if a.dim is None:
        est = MT.levina_bickel(X, a.k_eps) if a.dim_estimator == 'levina-bickel' else MT.twonn(X)
        d = int(np.clip(round(est), 1, min(a.max_dim, m - 1)))
        print(f'  n = {n}, m = {m}, {a.dim_estimator}: {est:.2f} → d = {d}, grupos = {c}')
    else:
        d = int(min(a.dim, m - 1)); print(f'  n = {n}, m = {m}, d = {d} (fornecido), grupos = {c}')
    mds_dims = a.mds_dims or max(10, d)
    weights = [w for w in a.weights if not (w == 'gdt_full' and m > a.full_max_dim)]
    if len(weights) < len(a.weights):
        print(f'  gdt_full omitido (m = {m} > --full-max-dim {a.full_max_dim}); use --pca para reduzir m')

    print('[2/4] grafo k-NN e métricas locais')
    A = MT.knn_graph(X, a.k, 'distance')
    nc, _ = connected_components(A)
    if nc > 1:
        A, added = MT.mst_bridges(X, A)
        print(f'  grafo {a.k}-NN com {nc} componentes: {added} aresta(s) da MST adicionada(s)')
    A = A.tocoo(); i, j = A.row, A.col
    L = {}
    if 'euclid' in weights:
        L['euclid'] = A.data
    if 'zmp' in weights:
        L['zmp'] = zmp_lengths(X, i, j, a.k_zmp)
    if {'tangent', 'sigma'} & set(weights):
        U, c_t, c_s = local_pca_ops(X, d, a.k)
        if 'tangent' in weights: L['tangent'] = edge_lengths(X, i, j, U, c_t)
        if 'sigma' in weights: L['sigma'] = edge_lengths(X, i, j, U, c_s)
        del U
    if 'gdt' in weights:
        U, cg = gdt_lowrank(X, d, a.k_graph, a.k_eps, a.k_smooth, a.eps_scale, a.h_floor,
                            max_support=a.max_support)
        L['gdt'] = edge_lengths(X, i, j, U, cg); del U
    if 'gdt_full' in weights:
        Uf, cf = gdt_lowrank(X, m, a.k_graph, a.k_eps, a.k_smooth, a.eps_scale, a.h_floor,
                             max_support=a.max_support, verbose=False)
        L['gdt_full'] = edge_lengths(X, i, j, Uf, cf); del Uf

    print('[3/4] geodésicas e clustering hierárquico')
    rows, labels = [], {}
    metrics = metric_set(a.mi_variant)
    for w in weights:
        D = geodesics(A, L[w])
        for lk in a.linkages:
            lab = cluster(D, lk, c, mds_dims, a.seed)
            labels[f'{w}__{lk}'] = lab
            sizes = np.sort(np.bincount(lab))[::-1]
            row = dict(peso=w, linkage=lk, n_grupos=len(sizes), maior_grupo=int(sizes[0]),
                       menor_grupo=int(sizes[-1]))
            if has_y:
                for mt, fn, _ in metrics:
                    row[mt] = float(fn(y, lab))
            rows.append(row)
        del D

    print('[4/4] resultados')
    prefix = a.out or f'hclust_{a.source}_{name}'.replace('/', '_').replace(' ', '_')
    keys = list(rows[0].keys())
    with open(f'{prefix}_results.csv', 'w', newline='') as fh:
        wcsv = csv.DictWriter(fh, fieldnames=keys); wcsv.writeheader(); wcsv.writerows(rows)
    if has_y:
        title = f'{name} ($n={n}$, $m={m}$, $d={d}$, {c} grupos)'.replace('_', r'\_')
        print(write_metric_tables(prefix, rows, weights, a.linkages, metrics, title))
        fig, axs = plt.subplots(2, 2, figsize=(6 + 4.4 * len(a.linkages), 10))
        x = np.arange(len(a.linkages)); wd = 0.8 / len(weights)
        for ax, (mt, _, desc) in zip(axs.ravel(), metrics):
            for k_, w in enumerate(weights):
                ax.bar(x + (k_ - (len(weights) - 1) / 2) * wd,
                       [next(r[mt] for r in rows if r['peso'] == w and r['linkage'] == lk) for lk in a.linkages],
                       wd, label=LABEL[w])
            ax.set_xticks(x); ax.set_xticklabels(a.linkages); ax.set_ylabel(mt)
            if mt != 'MI':
                ax.set_ylim(min(0, ax.get_ylim()[0]), 1.05)
            ax.set_title(desc); ax.grid(axis='y', alpha=0.3)
        axs[0, 0].legend(fontsize=8, ncol=2)
        fig.suptitle(f'{name} (n={n}, m={m}, d={d}, grupos={c}) — geodésicas no espaço original', fontsize=13)
        plt.tight_layout(); plt.savefig(f'{prefix}_metrics.png', dpi=120); plt.close(fig)
        print(f'\n  tabelas: ' + ', '.join(f"{prefix}_{mt.replace('-', '')}.csv" for mt, _, _ in metrics))
        print(f'  LaTeX:   {prefix}_tables.tex\n  gráfico: {prefix}_metrics.png')
    else:
        print('\n  sem rótulos: tamanho do maior grupo — linhas: linkage, colunas: peso')
        print(f'  {"":10s}' + ''.join(f'{LABEL[w]:>14s}' for w in weights))
        for lk in a.linkages:
            print(f'  {lk:10s}' + ''.join(
                f"{next(r['maior_grupo'] for r in rows if r['peso'] == w and r['linkage'] == lk):14d}" for w in weights))
    if a.save_labels:
        np.savez_compressed(f'{prefix}_labels.npz', y=y if has_y else [], **labels)
        print(f'  rótulos: {prefix}_labels.npz')
    print(f'  formato longo: {prefix}_results.csv\nconcluído em {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
