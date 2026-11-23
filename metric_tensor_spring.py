#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metric_tensor_spring.py
=======================

Tensor métrico local (GDT-h, Geodesic Diffusion Tensor) em cada amostra de um conjunto
de dados, visualizado como mapa de calor sobre o grafo k-NN desenhado com spring layout
no espaço original.

Pipeline
--------
1. Carrega um dataset do scikit-learn (load_*/fetch_*/make_*) ou do OpenML.
2. Pré-processa: one-hot de colunas categóricas, imputação pela mediana, remoção de
   colunas constantes, padronização (z-score), subamostragem opcional.
3. Estima a dimensão intrínseca d (Levina–Bickel ou TwoNN), ou usa --dim.
4. Calcula φ = Isomap com d componentes (coordenadas de dimensão intrínseca).
5. Estima G_i (d×d) com o GDT-h:
     - kernel gaussiano de banda fixa ε = c·mediana(r_k²), com peso pela distância
       euclidiana e suporte limitado pela distância geodésica no grafo k-NN (≤ 3√ε);
     - normalização α = 1 (Laplace–Beltrami), sem o próprio ponto na normalização;
     - tensor dual h_i = (2/ε) Σ_j P_ij Δφ_ij Δφ_ijᵀ, suavizado no grafo k-NN;
     - G_i = pseudoinversa de posto d de h_i.
6. Calcula escalares de G_i (padrão: tr G⁻¹ / d) e plota no spring layout do grafo k-NN
   construído no espaço original, com saturação dos maiores valores (--saturation).

Dependências
------------
    pip install numpy scipy scikit-learn matplotlib networkx pandas

Exemplos
--------
    python metric_tensor_spring.py --dataset digits --saturation 0.05
    python metric_tensor_spring.py --dataset digits --saturation 0.10 --scalar trGinv logdet aniso
    python metric_tensor_spring.py --dataset wine
    python metric_tensor_spring.py --dataset swiss_roll --n-samples 2000 --dim 2
    python metric_tensor_spring.py --source openml --dataset mnist_784 --max-samples 3000
    python metric_tensor_spring.py --source openml --data-id 61            # iris no OpenML
    python metric_tensor_spring.py --dataset digits --k-isomap 5 --mst      # Isomap com grafo k-NN ∪ MST
    python metric_tensor_spring.py --source openml --dataset "Fashion-MNIST" --max-samples 2500 --dim 8

Escalares disponíveis (--scalar)
--------------------------------
    trGinv    tr G⁻¹ / d        energia de Dirichlet de φ (estiramento local do embedding)  [padrão]
    logdet    ½ log det G       ampliação de volume (log)
    trG       tr G / d          estiramento quadrático médio
    aniso     log(λmax/λmin)    anisotropia (defeito conforme)
    mips      tr G / det G^{1/d}  energia MIPS (distorção conforme sem escala)
    dirichlet tr G + tr G⁻¹     Dirichlet simétrica
    isometry  ‖log G‖_F         defeito de isometria (após normalização global de escala)
    all       todos os anteriores (um arquivo por escalar)

Observações
-----------
- --mst: o grafo do Isomap passa a ser k-NN ∪ árvore geradora mínima (MST) das distâncias
  euclidianas. A MST conecta todos os pontos com as arestas mais curtas possíveis, então o
  grafo fica conexo para qualquer k, sem criar atalhos longos arbitrários. As geodésicas são
  calculadas nesse grafo e o embedding é obtido por MDS clássico (Isomap padrão).
- Custo de memória O(n²) (distâncias geodésicas densas): use --max-samples (padrão 3000).
- O spring layout é O(n²) por iteração: para n ≈ 3000 leva cerca de 1 minuto.
- G descreve a distorção do Isomap em relação às distâncias originais (padronizadas).
  Interprete os escalares como propriedades do embedding φ, não como curvatura.
"""
import argparse
import sys
import time
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import sparse
from scipy.sparse.csgraph import dijkstra, connected_components, minimum_spanning_tree, shortest_path
from scipy.linalg import eigh as _eigh
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=sparse.SparseEfficiencyWarning)

SCALARS = ['trGinv', 'logdet', 'trG', 'aniso', 'mips', 'dirichlet', 'isometry']
SCALAR_LABEL = {
    'trGinv':    'tr G⁻¹ / d  (energia de Dirichlet de φ)',
    'logdet':    '½ log det G  (ampliação de volume, log)',
    'trG':       'tr G / d  (estiramento quadrático médio)',
    'aniso':     'log(λmax / λmin)  (anisotropia)',
    'mips':      'tr G / (det G)^{1/d}  (energia MIPS)',
    'dirichlet': 'tr G + tr G⁻¹  (Dirichlet simétrica)',
    'isometry':  '‖log G‖_F  (defeito de isometria)',
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Carregamento e pré-processamento
# ══════════════════════════════════════════════════════════════════════════════
def load_sklearn(name, n_samples, seed):
    import sklearn.datasets as skd
    for prefix in ('load_', 'fetch_'):
        fn = getattr(skd, prefix + name, None)
        if fn is not None:
            out = fn()
            return out.data, getattr(out, 'target', None)
    fn = getattr(skd, 'make_' + name, None)
    if fn is not None:
        out = fn(n_samples=n_samples, random_state=seed)
        X, t = (out if isinstance(out, tuple) else (out, None))
        return X, t
    raise ValueError(f'Dataset "{name}" não encontrado no scikit-learn '
                     f'(tentei load_{name}, fetch_{name}, make_{name}).')


def load_openml(name, data_id, version):
    from sklearn.datasets import fetch_openml
    kw = dict(as_frame=True, parser='auto')
    if data_id is not None:
        out = fetch_openml(data_id=data_id, **kw)
    else:
        out = fetch_openml(name=name, version=version if version else 'active', **kw)
    return out.data, out.target


def to_numeric_matrix(X):
    """DataFrame/array → matriz float: one-hot de categóricas, imputação pela mediana."""
    import pandas as pd
    if isinstance(X, pd.DataFrame):
        cat = X.select_dtypes(exclude=[np.number, 'bool']).columns
        if len(cat):
            print(f'  one-hot em {len(cat)} coluna(s) categórica(s)')
            X = pd.get_dummies(X, columns=list(cat), dummy_na=False)
        X = X.astype(float).to_numpy()
    else:
        X = np.asarray(X, dtype=float)
    if X.ndim > 2:
        X = X.reshape(len(X), -1)                       # ex.: imagens (n, h, w)
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0)
        r, c = np.where(np.isnan(X)); X[r, c] = med[c]
        X[np.isnan(X)] = 0.0
    return X


def prepare(X, y, max_samples, seed, standardize=True, pca=None, float32=False):
    rng = np.random.default_rng(seed)
    X = to_numeric_matrix(X)
    if y is not None:
        y = np.asarray(y)
    if max_samples and len(X) > max_samples:
        if y is not None and len(np.unique(y)) <= 100:     # subamostragem estratificada
            from sklearn.model_selection import train_test_split
            idx, _ = train_test_split(np.arange(len(X)), train_size=max_samples,
                                      stratify=y, random_state=seed)
            idx = np.sort(idx)
        else:
            idx = np.sort(rng.choice(len(X), max_samples, replace=False))
        X = X[idx]; y = y[idx] if y is not None else None
        print(f'  subamostragem: {len(X)} amostras')
    keep = X.std(0) > 1e-12
    X = X[:, keep]
    if standardize:
        X = StandardScaler().fit_transform(X)
    if pca:
        from sklearn.decomposition import PCA
        p = PCA(n_components=pca if pca < 1 else int(pca), svd_solver='full' if pca < 1 else 'randomized',
                random_state=seed)
        m0 = X.shape[1]
        X = p.fit_transform(X)
        print(f'  PCA: {m0} → {X.shape[1]} componentes '
              f'({100 * p.explained_variance_ratio_.sum():.1f}% da variância)')
    if float32:
        X = X.astype(np.float32)
    # remove duplicatas exatas (distância zero quebra o k-NN e o kernel)
    keys = [hash(r.tobytes()) for r in np.ascontiguousarray(X.round(10))]
    _, uniq = np.unique(np.array(keys), return_index=True)
    if len(uniq) < len(X):
        uniq = np.sort(uniq)
        print(f'  {len(X) - len(uniq)} amostras duplicadas removidas')
        X = X[uniq]; y = y[uniq] if y is not None else None
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dimensão intrínseca
# ══════════════════════════════════════════════════════════════════════════════
def levina_bickel(X, k=15):
    dist = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=True)[0]
    dist = np.maximum(dist, 1e-12)
    return 1.0 / np.mean(np.log(dist[:, -1:] / dist[:, :-1]).mean(axis=1))


def twonn(X, discard=0.1):
    dist = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(return_distance=True)[0]
    mu = np.sort(dist[:, 1] / np.maximum(dist[:, 0], 1e-12))
    n = len(mu); mu = mu[: int(n * (1 - discard))]
    F = np.arange(1, len(mu) + 1) / n
    x, yv = np.log(mu), -np.log(1 - F)
    return (x @ yv) / (x @ x)


# ══════════════════════════════════════════════════════════════════════════════
# 2b. Isomap (com opção de grafo k-NN ∪ MST)
# ══════════════════════════════════════════════════════════════════════════════
def mst_bridges(X, A):
    """Arestas da MST entre componentes do grafo A (Borůvka): a cada rodada, cada componente
    ganha sua aresta mais curta para fora. Memória O(n·m), sem matriz de distâncias n×n.
    Dentro de uma componente conexa do k-NN, a MST não altera as geodésicas de forma relevante;
    o que garante a conexidade são justamente essas pontes mínimas."""
    A = A.tocsr().copy()
    added = 0
    while True:
        nc, lab = connected_components(A, directed=False)
        if nc == 1:
            return A, added
        best = {}
        for c in range(nc):
            inside = np.where(lab == c)[0]; outside = np.where(lab != c)[0]
            nn = NearestNeighbors(n_neighbors=1).fit(X[outside])
            dist, idx = nn.kneighbors(X[inside])
            a = int(np.argmin(dist[:, 0]))
            best[c] = (float(dist[a, 0]), int(inside[a]), int(outside[idx[a, 0]]))
        A = A.tolil()
        for w, u, v in set(best.values()):
            if A[u, v] == 0:
                A[u, v] = A[v, u] = max(w, 1e-12); added += 1
        A = A.tocsr()


def classical_mds(Dg, d):
    """MDS clássico com dupla centralização in-place (sem a matriz J n×n)."""
    B = Dg ** 2
    B -= B.mean(0, keepdims=True); B -= B.mean(1, keepdims=True)       # B.mean global já é 0
    B *= -0.5
    n = len(B)
    w, V = _eigh(B, subset_by_index=[n - d, n - 1], overwrite_a=True, check_finite=False)
    w, V = w[::-1], V[:, ::-1]
    if (w <= 0).any():
        print(f'  aviso: {int((w <= 0).sum())} autovalor(es) não positivo(s) no MDS')
    return V * np.sqrt(np.clip(w, 0, None))


def isomap_embedding(X, d, k, use_mst=False, landmarks=0, seed=0):
    """Isomap: geodésicas no grafo k-NN (opcionalmente + pontes da MST) + MDS clássico.
    landmarks > 0: Landmark Isomap (de Silva & Tenenbaum) — memória O(L·n) em vez de O(n²)."""
    A = knn_graph(X, k, 'distance')
    ncomp, _ = connected_components(A)
    n_new = 0
    if use_mst:
        A, n_new = mst_bridges(X, A)
        print(f'  grafo {k}-NN: {ncomp} componente(s) → 1 componente '
              f'({n_new} aresta(s) da MST adicionada(s))')
    elif ncomp > 1:
        print(f'  aviso: grafo {k}-NN tem {ncomp} componentes; use --mst ou aumente --k-isomap.')
        if not landmarks:
            return Isomap(n_neighbors=k, n_components=d).fit_transform(X), ncomp, 0
    if not landmarks:
        if not use_mst:
            return Isomap(n_neighbors=k, n_components=d).fit_transform(X), ncomp, 0
        return classical_mds(shortest_path(A, method='D', directed=False), d), ncomp, n_new
    # ── Landmark Isomap ──
    n = len(X)
    L = np.sort(np.random.default_rng(seed).choice(n, min(landmarks, n), replace=False))
    DL = dijkstra(A, directed=False, indices=L)                       # (L, n)
    if not np.isfinite(DL).all():
        raise RuntimeError('grafo desconexo: use --mst com --landmarks')
    DLL2 = DL[:, L] ** 2
    Y_L = classical_mds(DL[:, L], d)                                   # (L, d)
    mean_col = DLL2.mean(1)                                            # δ_μ
    pinvY = np.linalg.pinv(Y_L)                                        # (d, L)
    return (-0.5 * pinvY @ (DL ** 2 - mean_col[:, None])).T, ncomp, n_new


# ══════════════════════════════════════════════════════════════════════════════
# 3. GDT-h
# ══════════════════════════════════════════════════════════════════════════════
def knn_graph(X, k, mode='distance'):
    A = NearestNeighbors(n_neighbors=k).fit(X).kneighbors_graph(mode=mode)
    return A.maximum(A.T).tocsr()


def row_normalize(M):
    s = np.asarray(M.sum(1)).ravel()
    s[s == 0] = 1.0
    return (sparse.diags(1.0 / s) @ M).tocsr()


def weighted_moments(P, A):
    """h_i = Σ_j P_ij (A_j − A_i)(A_j − A_i)ᵀ, P estocástica por linha, sem materializar pares."""
    n, s = A.shape
    PAA = (P @ (A[:, :, None] * A[:, None, :]).reshape(n, s * s)).reshape(n, s, s)
    PA = P @ A
    return PAA - A[:, :, None] * PA[:, None, :] - PA[:, :, None] * A[:, None, :] \
        + A[:, :, None] * A[:, None, :]


def pinv_rank(H, d, floor=0.0):
    """Pseudoinversa de posto d. `floor`: piso relativo para os autovalores (fração da
    mediana global dos autovalores retidos) — evita G explosivo onde h é quase singular."""
    w, V = np.linalg.eigh(H)
    w, V = w[:, ::-1][:, :d], V[:, :, ::-1][:, :, :d]
    lo = floor * np.median(w) if floor > 0 else 1e-300
    n_floor = int((w < lo).any(1).sum())
    w = np.clip(w, lo, None)
    return np.einsum('nia,na,nja->nij', V, 1.0 / w, V), n_floor


def pair_sqdist(X, i, j, block=512):
    """‖x_i − x_j‖² só para os pares pedidos, em blocos de linhas (nunca materializa Δx).
    Memória: O(block · n), independente de m e do número de pares."""
    sq = np.einsum('ij,ij->i', X, X)
    out = np.empty(len(i), dtype=np.float64)
    order = np.argsort(i, kind='stable')
    i_s, j_s = i[order], j[order]
    starts = np.searchsorted(i_s, np.arange(0, len(X) + block, block))
    for b, r0 in enumerate(range(0, len(X), block)):
        a, z = starts[b], starts[b + 1]
        if a == z:
            continue
        rows = i_s[a:z] - r0
        Gram = X[r0:r0 + block] @ X.T                                 # (block, n)
        out[order[a:z]] = sq[i_s[a:z]] + sq[j_s[a:z]] - 2.0 * Gram[rows, j_s[a:z]]
    return np.maximum(out, 0.0)


def geodesic_support(A, radius, block=512):
    """Pares (i, j) com distância geodésica no grafo ≤ radius, como matriz esparsa.
    Dijkstra em blocos de origens: memória O(block · n) em vez de O(n²)."""
    n = A.shape[0]
    rows, cols = [], []
    for r0 in range(0, n, block):
        src = np.arange(r0, min(r0 + block, n))
        Dg = dijkstra(A, directed=False, indices=src, limit=radius)
        rr, cc = np.nonzero(np.isfinite(Dg) & (Dg > 0))
        rows.append(src[rr]); cols.append(cc)
    i = np.concatenate(rows); j = np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(i), bool), (i, j)), shape=(n, n))


def gdt_h(X, Phi, d, k_graph=10, k_eps=15, k_smooth=10, eps_scale=1.0, h_floor=1e-2,
          max_support=None, block=512, verbose=True):
    """Geodesic Diffusion Tensor (variante híbrida): retorna G (n, s, s) e diagnósticos.

    max_support: se definido, mantém por ponto só os `max_support` pares de suporte mais
    próximos (limita memória e tempo quando o suporte geodésico é muito grande)."""
    n = len(X)
    r_k = NearestNeighbors(n_neighbors=k_eps).fit(X).kneighbors(return_distance=True)[0][:, -1]
    eps = eps_scale * np.median(r_k) ** 2
    A = knn_graph(X, k_graph, 'distance')
    Sup = geodesic_support(A, 3.0 * np.sqrt(eps), block)
    Sup = (Sup + (A > 0)).tocsr()                                       # garante os k vizinhos
    Sup.setdiag(False); Sup.eliminate_zeros()
    i, j = Sup.nonzero()
    chord2 = pair_sqdist(X, i, j, block)                                # peso pela corda
    if max_support:                                                     # poda por ponto
        keep = np.zeros(len(i), bool)
        indptr = np.concatenate([[0], np.cumsum(np.bincount(i, minlength=n))])
        for a in range(n):
            lo, hi = indptr[a], indptr[a + 1]
            if hi - lo <= max_support:
                keep[lo:hi] = True
            else:
                keep[lo + np.argpartition(chord2[lo:hi], max_support)[:max_support]] = True
        i, j, chord2 = i[keep], j[keep], chord2[keep]
    K = sparse.csr_matrix((np.exp(-chord2 / eps), (i, j)), shape=(n, n))
    K = K.maximum(K.T)                                                  # simetria após a poda
    q = np.asarray(K.sum(1)).ravel() + 1.0                              # densidade (com o ponto)
    K1 = sparse.diags(1.0 / q) @ K @ sparse.diags(1.0 / q)              # α = 1
    P = row_normalize(K1)                                               # sem o próprio ponto
    h = (2.0 / eps) * weighted_moments(P, Phi)
    S = row_normalize(knn_graph(X, k_smooth, 'connectivity') + sparse.identity(n))
    h = (S @ h.reshape(n, -1)).reshape(h.shape)                         # suavização no grafo
    G, n_floor = pinv_rank(h, d, h_floor)
    neff = 1.0 / np.asarray(P.multiply(P).sum(1)).ravel()
    if verbose:
        print(f'  ε = {eps:.4g}  |  pares no suporte (mediana por ponto) = '
              f'{np.median(np.diff(K.indptr)):.0f}  |  N_eff do kernel (mediana) = {np.median(neff):.1f}')
        if n_floor:
            print(f'  {n_floor} ponto(s) com autovalor de h abaixo do piso ({h_floor:g} × mediana): '
                  f'regularizados (h quase singular, poucas amostras efetivas)')
            if n_floor > 0.1 * n:
                print(f'  aviso: {100 * n_floor / n:.0f}% dos pontos precisaram de regularização — '
                      f'd = {d} pode estar acima da dimensão local; tente --dim {max(d - 1, 1)}')
    return G, dict(eps=eps, neff=neff)


def scalars_from_G(G, d):
    """Escalares do tensor métrico (após normalizar a escala típica para 1)."""
    w = np.linalg.eigvalsh(G)[:, -d:]                    # d autovalores não nulos
    w = np.clip(w, 1e-300, None)
    w = w / np.exp(np.median(np.log(w).mean(1)))
    lw = np.log(w)
    trinv = (1.0 / w).sum(1)
    return {
        'trGinv':    trinv / d,
        'logdet':    0.5 * lw.sum(1),
        'trG':       w.sum(1) / d,
        'aniso':     lw[:, -1] - lw[:, 0],
        'mips':      w.sum(1) / np.exp(lw.mean(1)),
        'dirichlet': w.sum(1) + trinv,
        'isometry':  np.linalg.norm(lw, axis=1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Visualização
# ══════════════════════════════════════════════════════════════════════════════
def spring_layout(X, k, seed, iterations):
    import networkx as nx
    nb = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    Gk = nx.Graph(); Gk.add_nodes_from(range(len(X)))
    Gk.add_edges_from((i, int(j)) for i in range(len(X)) for j in nb[i])
    pos = nx.spring_layout(Gk, seed=seed, iterations=iterations)
    return np.array([pos[i] for i in range(len(X))]), nb


def draw_graph(ax, L, nb, colors, cmap='jet', s=14):
    segs = [[L[i], L[j]] for i in range(len(L)) for j in nb[i]]
    ax.add_collection(LineCollection(segs, colors='0.75', linewidths=0.3, zorder=1))
    sc = ax.scatter(L[:, 0], L[:, 1], c=colors, cmap=cmap, vmin=0, vmax=1, s=s,
                    edgecolors='k', linewidths=0.15, zorder=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.autoscale_view()
    return sc


def saturate(v, upper, lower=0.0):
    """Normaliza em [0, 1] saturando a fração `upper` de maiores valores (e `lower` de menores)."""
    lo = np.quantile(v, lower) if lower > 0 else v.min()
    hi = np.quantile(v, 1.0 - upper) if upper > 0 else v.max()
    return np.clip((v - lo) / max(hi - lo, 1e-300), 0, 1), lo, hi


def plot(L, nb, v, y, name, title_ds, saturation, lower_saturation, out, point_size):
    v01, lo, hi = saturate(v, saturation, lower_saturation)
    has_y = y is not None
    fig, axs = plt.subplots(1, 2 if has_y else 1, figsize=(22 if has_y else 12, 10),
                            gridspec_kw=dict(width_ratios=[1, 1.08], wspace=0.05) if has_y else None)
    axs = np.atleast_1d(axs)
    if has_y:
        ax = axs[0]
        yy = np.asarray(y)
        try:
            yf = yy.astype(float); discrete = len(np.unique(yf)) <= 20
        except (TypeError, ValueError):
            discrete = True
        if discrete:
            cats, codes = np.unique(yy.astype(str), return_inverse=True)
            cm = plt.get_cmap('tab10' if len(cats) <= 10 else 'tab20')
            ax.scatter(L[:, 0], L[:, 1], c=codes, cmap=cm, s=point_size * 0.6,
                       vmin=-0.5, vmax=max(len(cats), 10 if len(cats) <= 10 else 20) - 0.5)
            for k, c in enumerate(cats):
                ax.text(*np.median(L[codes == k], 0), str(c), fontsize=13, weight='bold',
                        ha='center', va='center', bbox=dict(fc='w', alpha=0.7, lw=0))
            ax.set_title(f'{title_ds} — spring layout do grafo k-NN (classes)', fontsize=13)
        else:
            sc0 = ax.scatter(L[:, 0], L[:, 1], c=yf, cmap='viridis', s=point_size * 0.6)
            plt.colorbar(sc0, ax=ax, fraction=0.046, pad=0.02, label='alvo')
            ax.set_title(f'{title_ds} — spring layout do grafo k-NN (alvo contínuo)', fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
    ax = axs[-1]
    sc = draw_graph(ax, L, nb, v01, s=point_size)
    pct = f'{100 * saturation:g}% maiores valores saturados (≥ {hi:.3g})' if saturation > 0 else 'sem saturação'
    if lower_saturation > 0:
        pct += f'; {100 * lower_saturation:g}% menores (≤ {lo:.3g})'
    ax.set_title(f'{SCALAR_LABEL[name]}\n{pct}', fontsize=13)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    t = np.linspace(0, 1, 6); cb.set_ticks(t)
    lab = [f'{lo + s_ * (hi - lo):.3g}' for s_ in t]
    if saturation > 0: lab[-1] = '≥ ' + lab[-1]
    if lower_saturation > 0: lab[0] = '≤ ' + lab[0]
    cb.set_ticklabels(lab); cb.set_label(name)
    plt.savefig(out, dpi=110, bbox_inches='tight'); plt.close(fig)
    return lo, hi


# ══════════════════════════════════════════════════════════════════════════════
# 5. Programa principal
# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group('dados')
    g.add_argument('--source', choices=['sklearn', 'openml'], default='sklearn')
    g.add_argument('--dataset', default='digits', help='nome no scikit-learn ou no OpenML')
    g.add_argument('--data-id', type=int, default=None, help='id no OpenML (alternativa ao nome)')
    g.add_argument('--version', default=None, help='versão no OpenML (padrão: ativa)')
    g.add_argument('--n-samples', type=int, default=2000, help='para geradores make_* do scikit-learn')
    g.add_argument('--max-samples', type=int, default=3000, help='subamostragem (0 = todas)')
    g.add_argument('--no-standardize', action='store_true')
    g = p.add_argument_group('memória e desempenho (úteis para m ou n grandes)')
    g.add_argument('--pca', type=float, default=None,
                   help='reduz os atributos antes de tudo: inteiro = nº de componentes; '
                        'fração em (0,1) = variância retida (ex.: 0.99)')
    g.add_argument('--float32', action='store_true', help='armazena X em float32 (metade da memória)')
    g.add_argument('--block', type=int, default=512,
                   help='tamanho dos blocos de linhas (distâncias e Dijkstra); menor = menos memória')
    g.add_argument('--max-support', type=int, default=None,
                   help='máximo de pares no suporte do kernel por ponto (ex.: 200)')
    g.add_argument('--landmarks', type=int, default=0,
                   help='Landmark Isomap com L marcos (memória O(L·n)); 0 = Isomap completo')
    g = p.add_argument_group('geometria')
    g.add_argument('--dim', type=int, default=None, help='dimensão intrínseca (padrão: estimada)')
    g.add_argument('--dim-estimator', choices=['levina-bickel', 'twonn'], default='levina-bickel')
    g.add_argument('--max-dim', type=int, default=12, help='limite superior para d estimado')
    g.add_argument('--k-isomap', type=int, default=15)
    g.add_argument('--mst', action='store_true',
                   help='une a árvore geradora mínima ao grafo k-NN do Isomap (garante conexidade)')
    g.add_argument('--k-graph', type=int, default=10, help='k do grafo das geodésicas')
    g.add_argument('--k-eps', type=int, default=15, help='k usado para escolher ε')
    g.add_argument('--k-smooth', type=int, default=10, help='k do grafo de suavização de h')
    g.add_argument('--eps-scale', type=float, default=1.0, help='ε = eps_scale · mediana(r_k²)')
    g.add_argument('--h-floor', type=float, default=1e-2,
                   help='piso relativo dos autovalores de h antes de inverter (0 = desliga)')
    g = p.add_argument_group('visualização')
    g.add_argument('--scalar', nargs='+', default=['trGinv'], choices=SCALARS + ['all'])
    g.add_argument('--saturation', type=float, default=0.05,
                   help='fração de MAIORES valores saturados (ex.: 0.05 = 5%%, 0.10 = 10%%)')
    g.add_argument('--lower-saturation', type=float, default=0.0,
                   help='fração de MENORES valores saturados (padrão 0)')
    g.add_argument('--k-layout', type=int, default=15, help='k do grafo desenhado')
    g.add_argument('--layout-iterations', type=int, default=300)
    g.add_argument('--point-size', type=float, default=14)
    g.add_argument('--out', default=None, help='prefixo dos arquivos de saída')
    g.add_argument('--save-values', action='store_true', help='salva escalares e layout em .npz')
    g.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    if not 0 <= a.saturation < 1 or not 0 <= a.lower_saturation < 1:
        sys.exit('--saturation e --lower-saturation devem estar em [0, 1).')
    t0 = time.time()
    name = a.dataset if a.data_id is None else f'openml_{a.data_id}'
    print(f'[1/5] carregando {a.source}:{name}')
    if a.source == 'sklearn':
        X, y = load_sklearn(a.dataset, a.n_samples, a.seed)
    else:
        X, y = load_openml(a.dataset, a.data_id, a.version)
    X, y = prepare(X, y, a.max_samples, a.seed, standardize=not a.no_standardize,
                   pca=a.pca, float32=a.float32)
    n, m = X.shape
    print(f'  n = {n}, m = {m}')

    print('[2/5] dimensão intrínseca')
    if a.dim is None:
        est = levina_bickel(X, a.k_eps) if a.dim_estimator == 'levina-bickel' else twonn(X)
        d = int(np.clip(round(est), 1, min(a.max_dim, m - 1)))
        print(f'  {a.dim_estimator}: {est:.2f} → d = {d}')
    else:
        d = int(min(a.dim, m - 1)); print(f'  d = {d} (fornecido)')

    print(f'[3/5] Isomap com {d} componentes' + (' (grafo k-NN ∪ MST)' if a.mst else '')
          + (f' — Landmark Isomap com {a.landmarks} marcos' if a.landmarks else ''))
    Phi, _, _ = isomap_embedding(X, d, a.k_isomap, use_mst=a.mst, landmarks=a.landmarks, seed=a.seed)

    print('[4/5] tensor métrico GDT-h')
    G, info = gdt_h(X, Phi, d, a.k_graph, a.k_eps, a.k_smooth, a.eps_scale, a.h_floor,
                    max_support=a.max_support, block=a.block)
    S = scalars_from_G(G, d)

    print(f'[5/5] spring layout do grafo {a.k_layout}-NN no espaço original')
    L, nb = spring_layout(X, a.k_layout, a.seed, a.layout_iterations)

    names = SCALARS if 'all' in a.scalar else a.scalar
    prefix = a.out or (f'metric_{a.source}_{name}' + (f'_k{a.k_isomap}mst' if a.mst else '')
                       ).replace('/', '_').replace(' ', '_')
    title_ds = f'{name} (n={n}, m={m}, d={d})'
    for s_name in names:
        out = f'{prefix}_{s_name}_sat{100 * a.saturation:g}.png'
        lo, hi = plot(L, nb, S[s_name], y, s_name, title_ds, a.saturation, a.lower_saturation,
                      out, a.point_size)
        v = S[s_name]
        print(f'  {s_name:9s} min={v.min():.3g} mediana={np.median(v):.3g} max={v.max():.3g} '
              f'| escala de cores [{lo:.3g}, {hi:.3g}] → {out}')
    if a.save_values:
        np.savez_compressed(f'{prefix}_values.npz', layout=L, phi=Phi, G=G, y=y if y is not None else [],
                            **S, neff=info['neff'], eps=info['eps'])
        print(f'  valores salvos em {prefix}_values.npz')
    print(f'concluído em {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
