#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_hclust.py
===================

Compara o clustering hierárquico sobre geodésicas no grafo k-NN do espaço original para vários
pesos de aresta e ligações (ward, single, complete, average),
numa lista de conjuntos de dados, por QUATRO métricas de avaliação, e monta uma tabela por métrica:

  ARI         Adjusted Rand Index
  AMI         informação mútua ajustada (padrão; --mi-variant nmi ou mi para NMI ou MI bruta)
  FMI         índice de Fowlkes–Mallows
  V-measure   média harmônica de homogeneidade e completude

Pesos de aresta (--weights; padrão: euclid tangent zmp gdt gdt_full)
  euclid    ‖Δx‖
  tangent   ‖P_T Δx‖ (projeção no tangente da PCA local, posto d)
  zmp       ‖Δx‖ / √(σ_i σ_j), σ_i = distância ao K-ésimo vizinho (self-tuning de
            Zelnik-Manor & Perona, 2004; --k-zmp, padrão K = 7) — baseline de reescala local
  sigma     √(Δxᵀ Σ⁺ Δx), Σ = covariância local de posto d (opcional; não está no padrão)
  gdt       √(Δxᵀ ½(G_i + G_j) Δx), G = GDT de posto d
  gdt_full  idem, posto cheio (só quando m ≤ --full-max-dim)

Regras de pré-processamento (por conjunto):
  - colunas categóricas → one-hot; valores ausentes → mediana; colunas constantes removidas;
  - padronização (z-score);
  - se m > 100: PCA para 50 componentes (--pca-threshold / --pca-dims);
  - subamostragem estratificada para no máximo --max-samples amostras (padrão 10000).

Fontes dos dados:
  - nomes com loader no scikit-learn (ex.: 'digits') vêm do scikit-learn;
  - os demais vêm do OpenML por nome (fetch_openml);
  - --local NOME=ARQUIVO usa um arquivo local (.npz com X e y, ou .csv com o rótulo na
    última coluna) — útil quando o conjunto não está no OpenML com esse nome;
  - --openml-id NOME=ID força um id específico do OpenML.
Cada conjunto roda isolado: erros (nome não encontrado, falta de memória etc.) são
registrados e o conjunto é pulado. Com --resume, só são calculados os pesos que ainda não
estão no CSV para cada conjunto (ex.: rodar de novo com 'zmp' incluído calcula só o ZMP nos
conjuntos já concluídos, sem refazer euclid/tangent/gdt/gdt_full).

Saídas (prefixo --out, padrão 'benchmark'; <M> ∈ {ARI, AMI, FMI, Vmeasure}):
  <prefixo>_long.csv         uma linha por (conjunto, repetição, peso, ligação), com as 4 métricas
  <prefixo>_<M>_wide.csv     média da métrica: linhas = conjuntos; colunas = ligação × peso
  <prefixo>_<M>_table.tex    tabela LaTeX (booktabs) da métrica, melhor peso de cada ligação em
                             negrito, com média, rank médio e vitórias por peso
  <prefixo>_tables.tex       as quatro tabelas LaTeX num único arquivo
  <prefixo>_summary.txt      as quatro tabelas em texto
  <prefixo>_errors.txt       conjuntos que falharam e o motivo

Exemplos
--------
    python benchmark_hclust.py                                   # lista padrão completa
    python benchmark_hclust.py --datasets digits mfeat-zernike optdigits
    python benchmark_hclust.py --local coil-20=coil20.npz --resume
    python benchmark_hclust.py --max-samples 2000 --repeats 3
    python benchmark_hclust.py --max-support 200               # mais rápido
    python benchmark_hclust.py --resume                          # acrescenta o ZMP a um CSV antigo
    python benchmark_hclust.py --weights euclid zmp sigma gdt --k-zmp 7
"""
import argparse
import csv
import os
import sys
import time
import traceback
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metric_tensor_spring as MT                                # noqa: E402
import hierarchical_metric_clustering as H                      # noqa: E402
from scipy.sparse.csgraph import connected_components           # noqa: E402
from sklearn.decomposition import PCA                           # noqa: E402

warnings.filterwarnings('ignore')

# Os 22 conjuntos usados no artigo (scikit-learn: iris, digits; os demais via OpenML, pelo nome)
DATASETS = ['iris', 'penguins', 'user-knowledge', 'seeds', 'digits', 'mfeat-zernike', 'mfeat-karhunen',
            'mfeat-fourier', 'mfeat-factors', 'semeion', 'pendigits', 'coil-20', 'Olivetti_Faces', 'USPS',
            'artificial-characters', 'satimage', 'texture', 'UMIST_Faces_Cropped', 'letter', 'waveform-5000',
            'isolet', 'Indian_pines']
ALL_WEIGHTS = ['euclid', 'tangent', 'zmp', 'sigma', 'gdt', 'gdt_full']
WEIGHTS = ['euclid', 'tangent', 'zmp', 'gdt', 'gdt_full']                     # padrão
LINKAGES = ['ward', 'single', 'complete', 'average']
PAPER_LINKAGES = ['ward', 'complete', 'average']                               # usadas no artigo
WLABEL = {'euclid': 'Euclid', 'tangent': 'Tangente', 'zmp': 'ZMP', 'sigma': r'$\Sigma^{-1}$',
          'gdt': 'GDT', 'gdt_full': 'GDT$_m$'}


# ══════════════════════════════════════════════════════════════════════════════
# Carregamento
# ══════════════════════════════════════════════════════════════════════════════
def load_local(path):
    if path.endswith('.npz'):
        z = np.load(path, allow_pickle=True)
        X = z['X']; y = z['y'] if 'y' in z else z['label']
        return X.reshape(len(X), -1), y
    import pandas as pd
    df = pd.read_csv(path)
    return df.iloc[:, :-1], df.iloc[:, -1].to_numpy()


def load_dataset(name, local, openml_ids):
    import sklearn.datasets as skd
    if name in local:
        return load_local(local[name]), f'arquivo {local[name]}'
    if name in openml_ids:
        X, y = MT.load_openml(None, openml_ids[name], None)
        return (X, y), f'OpenML id {openml_ids[name]}'
    if hasattr(skd, f'load_{name}'):
        out = getattr(skd, f'load_{name}')()
        return (out.data, out.target), 'scikit-learn'
    X, y = MT.load_openml(name, None, None)
    return (X, y), 'OpenML'


def preprocess(X, y, max_samples, pca_threshold, pca_dims, seed):
    if y is None:
        raise ValueError('conjunto sem rótulo (target) — necessário para o ARI')
    y = np.asarray(y)
    keep = ~np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in y])
    X = X[keep] if not hasattr(X, 'iloc') else X.iloc[np.where(keep)[0]]
    y = y[keep]
    m_orig = X.shape[1] if X.ndim == 2 else int(np.prod(X.shape[1:]))
    X, y = MT.prepare(X, y, max_samples, seed, standardize=True)
    pca_info = ''
    if X.shape[1] > pca_threshold:
        p = PCA(n_components=pca_dims, svd_solver='randomized', random_state=seed)
        m0 = X.shape[1]; X = p.fit_transform(X)
        pca_info = f'PCA {m0}→{pca_dims} ({100 * p.explained_variance_ratio_.sum():.0f}% var.)'
    y = np.unique(np.asarray(y).astype(str), return_inverse=True)[1]
    return X, y, m_orig, pca_info


# ══════════════════════════════════════════════════════════════════════════════
# Um conjunto, uma repetição
# ══════════════════════════════════════════════════════════════════════════════
def run_one(X, y, args, weights, linkages, metrics):
    n, m = X.shape
    c = len(np.unique(y))
    est = MT.levina_bickel(X, args.k_eps)
    d = int(np.clip(round(est), 1, min(args.max_dim, m - 1)))
    A = MT.knn_graph(X, args.k, 'distance')
    nc, _ = connected_components(A)
    if nc > 1:
        A, _ = MT.mst_bridges(X, A)
    A = A.tocoo(); i, j = A.row, A.col
    L, tw = {}, {}
    t = time.time()
    if 'euclid' in weights:
        L['euclid'] = A.data; tw['euclid'] = 0.0
    if 'zmp' in weights:
        t = time.time()
        L['zmp'] = H.zmp_lengths(X, i, j, args.k_zmp)
        tw['zmp'] = time.time() - t
    if {'tangent', 'sigma'} & set(weights):
        t = time.time()
        U, c_t, c_s = H.local_pca_ops(X, d, args.k)
        t_pca = time.time() - t
        for w_, c_ in (('tangent', c_t), ('sigma', c_s)):
            if w_ in weights:
                t = time.time()
                L[w_] = H.edge_lengths(X, i, j, U, c_)
                tw[w_] = t_pca + time.time() - t
        del U
    if 'gdt' in weights:
        t = time.time()
        U, cg = H.gdt_lowrank(X, d, args.k_graph, args.k_eps, args.k_smooth, args.eps_scale,
                              args.h_floor, max_support=args.max_support, verbose=False)
        L['gdt'] = H.edge_lengths(X, i, j, U, cg); del U
        tw['gdt'] = time.time() - t
    if 'gdt_full' in weights:
        if m > args.full_max_dim:
            print(f'      gdt_full omitido (m = {m} > {args.full_max_dim})')
        else:
            t = time.time()
            U, cf = H.gdt_lowrank(X, m, args.k_graph, args.k_eps, args.k_smooth, args.eps_scale,
                                  args.h_floor, max_support=args.max_support, verbose=False)
            L['gdt_full'] = H.edge_lengths(X, i, j, U, cf); del U
            tw['gdt_full'] = time.time() - t
    rows = []
    for w, lw in L.items():
        D = H.geodesics(A, lw)
        for lk in linkages:
            t = time.time()
            lab = H.cluster(D, lk, c, args.mds_dims or max(10, d), args.seed)
            row = dict(weight=w, linkage=lk, d=d, d_est=round(est, 2), k_clusters=c,
                       t_metric=round(tw[w], 2), t_cluster=round(time.time() - t, 2))
            for mt, fn, _ in metrics:
                row[mt] = float(fn(y, lab))
            rows.append(row)
        del D
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Tabelas
# ══════════════════════════════════════════════════════════════════════════════
def aggregate(long_rows, datasets, weights, linkages, metric):
    """Média e desvio da métrica por (conjunto, ligação, peso)."""
    agg = {}
    for ds in datasets:
        for lk in linkages:
            for w in weights:
                v = [float(r[metric]) for r in long_rows
                     if r['dataset'] == ds and r['linkage'] == lk and r['weight'] == w]
                if v:
                    agg[(ds, lk, w)] = (float(np.mean(v)), float(np.std(v)), len(v))
    return agg


def ranks_and_wins(agg, datasets, weights, linkages):
    from scipy.stats import rankdata
    out = {}
    for lk in linkages:
        R, wins, means = {w: [] for w in weights}, {w: 0 for w in weights}, {w: [] for w in weights}
        for ds in datasets:
            ws = [w for w in weights if (ds, lk, w) in agg]
            if len(ws) < 2:
                continue
            vals = np.round(np.array([agg[(ds, lk, w)][0] for w in ws]), 3)   # empates na precisão exibida
            rk = rankdata(-vals)                                     # 1 = melhor
            for w, r_, v in zip(ws, rk, vals):
                R[w].append(r_); means[w].append(v)
            best = vals.max()
            for w, v in zip(ws, vals):
                if v >= best - 1e-9:
                    wins[w] += 1
        out[lk] = {w: (np.mean(means[w]) if means[w] else np.nan,
                       np.mean(R[w]) if R[w] else np.nan, wins[w], len(R[w])) for w in weights}
    return out


def write_metric_table(prefix, long_rows, meta, datasets, weights, linkages, metric, desc):
    cap = desc[:1].upper() + desc[1:]                              # legenda com inicial maiúscula
    done = [ds for ds in datasets if any(r['dataset'] == ds for r in long_rows)]
    agg = aggregate(long_rows, done, weights, linkages, metric)
    tag = metric.replace('-', '')
    summ = ranks_and_wins(agg, done, weights, linkages)
    # ── wide CSV ──
    with open(f'{prefix}_{tag}_wide.csv', 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['dataset', 'n', 'm_original', 'm_usado', 'd'] + [f'{lk}_{w}' for lk in linkages for w in weights])
        for ds in done:
            mt = meta[ds]
            wr.writerow([ds, mt['n'], mt['m_orig'], mt['m'], mt['d']] +
                        [f'{agg[(ds, lk, w)][0]:.4f}' if (ds, lk, w) in agg else '' for lk in linkages for w in weights])
    # ── LaTeX ──
    ncol = len(linkages) * len(weights)
    L = [r'\begin{table*}[t]', r'\centering',
         rf'\caption{{{cap} do clustering hierárquico sobre geodésicas no grafo $k$-NN do espaço original, '
         r'com diferentes pesos de aresta. Em negrito, o melhor peso para cada ligação. '
         r'Conjuntos com $m>100$ foram reduzidos a 50 componentes por PCA.}',
         rf'\label{{tab:hclust-{tag.lower()}}}', r'\resizebox{\textwidth}{!}{%',
         r'\begin{tabular}{l' + 'c' * ncol + '}', r'\toprule',
         ' & ' + ' & '.join(rf'\multicolumn{{{len(weights)}}}{{c}}{{{lk.capitalize()}}}' for lk in linkages) + r' \\',
         ''.join(rf'\cmidrule(lr){{{2 + k * len(weights)}-{1 + (k + 1) * len(weights)}}}' for k in range(len(linkages))),
         'Conjunto & ' + ' & '.join(WLABEL[w] for _ in linkages for w in weights) + r' \\', r'\midrule']
    for ds in done:
        cells = []
        for lk in linkages:
            vals = [round(agg[(ds, lk, w)][0], 3) if (ds, lk, w) in agg else np.nan for w in weights]
            best = np.nanmax(vals) if not np.all(np.isnan(vals)) else np.nan
            for v in vals:
                if np.isnan(v):
                    cells.append('--')
                else:
                    s = f'{v:.3f}'
                    cells.append(rf'\textbf{{{s}}}' if abs(v - best) < 1e-9 else s)
        nm = ds.replace('_', r'\_')
        L.append(f'{nm} & ' + ' & '.join(cells) + r' \\')
    L.append(r'\midrule')
    for label, idx, fmt in ((f'{metric} médio', 0, '{:.3f}'), ('Rank médio', 1, '{:.2f}'), ('Vitórias', 2, '{:d}')):
        cells = []
        for lk in linkages:
            col = [summ[lk][w][idx] for w in weights]
            if idx == 1:
                best = np.nanmin(col)
            else:
                best = np.nanmax(col)
            for v in col:
                if isinstance(v, float) and np.isnan(v):
                    cells.append('--'); continue
                s = fmt.format(int(v) if idx == 2 else v)
                cells.append(rf'\textbf{{{s}}}' if abs(v - best) < 1e-9 else s)
        L.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}']
    tex = '\n'.join(L) + '\n'
    with open(f'{prefix}_{tag}_table.tex', 'w') as fh:
        fh.write('% requer \\usepackage{booktabs} e \\usepackage{graphicx}\n' + tex)
    # ── resumo em texto ──
    lines = [f'══ {metric} — {desc} ══']
    lines.append(f'{"":22s}' + ''.join(f'{lk:^40s}' for lk in linkages))
    lines.append(f'{"conjunto":22s}' + ''.join(''.join(f'{w:>10s}' for w in weights) for _ in linkages))
    for ds in done:
        lines.append(f'{ds[:22]:22s}' + ''.join(''.join(
            f'{agg[(ds, lk, w)][0]:10.3f}' if (ds, lk, w) in agg else f'{"--":>10s}' for w in weights) for lk in linkages))
    for label, idx in ((f'{metric} médio', 0), ('rank médio', 1), ('vitórias', 2)):
        lines.append(f'{label:22s}' + ''.join(''.join(
            (f'{summ[lk][w][idx]:10.3f}' if idx < 2 else f'{summ[lk][w][idx]:10d}') for w in weights) for lk in linkages))
    return '\n'.join(lines), tex


def write_tables(prefix, long_rows, meta, datasets, weights, linkages, metrics):
    """Gera as tabelas das quatro métricas (CSV largo, LaTeX e texto)."""
    txts, texs = [], []
    for mt, _, desc in metrics:
        t, x = write_metric_table(prefix, long_rows, meta, datasets, weights, linkages, mt, desc)
        txts.append(t); texs.append(x)
    with open(f'{prefix}_tables.tex', 'w') as fh:
        fh.write('% requer \\usepackage{booktabs} e \\usepackage{graphicx}\n' + '\n'.join(texs))
    txt = '\n\n'.join(txts)
    with open(f'{prefix}_summary.txt', 'w') as fh:
        fh.write(txt + '\n')
    return txt


# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', nargs='+', default=DATASETS)
    p.add_argument('--weights', nargs='+', default=WEIGHTS, choices=ALL_WEIGHTS)
    p.add_argument('--linkages', nargs='+', default=PAPER_LINKAGES, choices=LINKAGES,
                   help='padrão: ward complete average (artigo); acrescente single se desejar')
    p.add_argument('--local', nargs='*', default=[], help='NOME=ARQUIVO (.npz com X e y, ou .csv)')
    p.add_argument('--openml-id', nargs='*', default=[], help='NOME=ID para forçar um id do OpenML')
    p.add_argument('--max-samples', type=int, default=10000)
    p.add_argument('--repeats', type=int, default=1, help='subamostras independentes (média ± dp)')
    p.add_argument('--pca-threshold', type=int, default=100)
    p.add_argument('--pca-dims', type=int, default=50)
    p.add_argument('--max-dim', type=int, default=10)
    p.add_argument('--k', type=int, default=15)
    p.add_argument('--k-graph', type=int, default=10)
    p.add_argument('--k-eps', type=int, default=15)
    p.add_argument('--k-smooth', type=int, default=10)
    p.add_argument('--eps-scale', type=float, default=1.0)
    p.add_argument('--h-floor', type=float, default=1e-2)
    p.add_argument('--max-support', type=int, default=None,
                   help='limita os pares do kernel do GDT por ponto (ex.: 200); acelera, muda pouco o resultado')
    p.add_argument('--full-max-dim', type=int, default=100)
    p.add_argument('--k-zmp', type=int, default=7,
                   help='K do self-tuning ZMP: σ_i = distância ao K-ésimo vizinho (padrão 7, como no artigo)')
    p.add_argument('--mds-dims', type=int, default=None)
    p.add_argument('--mi-variant', choices=['ami', 'nmi', 'mi'], default='ami',
                   help='informação mútua: ajustada (padrão), normalizada ou bruta')
    p.add_argument('--out', default='benchmark')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    metrics = H.metric_set(args.mi_variant)
    mnames = [mt for mt, _, _ in metrics]
    datasets = list(dict.fromkeys(args.datasets))                  # remove nomes repetidos
    if len(datasets) < len(args.datasets):
        rep_ = sorted({d_ for d_ in args.datasets if args.datasets.count(d_) > 1})
        print(f'aviso: conjunto(s) repetido(s) na lista removido(s): {", ".join(rep_)}')
    local = dict(s.split('=', 1) for s in args.local)
    oml = {k: int(v) for k, v in (s.split('=', 1) for s in args.openml_id)}
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)      # cria a pasta de saída
    long_csv = f'{args.out}_long.csv'
    fields = (['dataset', 'repeat', 'source', 'n', 'm_original', 'm_used', 'preproc', 'weight', 'linkage']
              + mnames + ['d', 'd_est', 'k_clusters', 't_metric', 't_cluster'])
    long_rows, meta = [], {}
    if args.resume and os.path.exists(long_csv):
        with open(long_csv) as fh:
            rd = csv.DictReader(fh)
            header = rd.fieldnames or []
            if not set(mnames) <= set(header):                       # CSV de versão antiga
                old = long_csv.replace('.csv', '_antigo.csv')
                print(f'aviso: {long_csv} não tem as colunas {mnames}; renomeado para {old} '
                      f'e todos os conjuntos serão recalculados')
                fh.close(); os.replace(long_csv, old)
                with open(long_csv, 'w', newline='') as fo:
                    csv.DictWriter(fo, fieldnames=fields).writeheader()
            else:
                for r in rd:
                    for mt in mnames:
                        r[mt] = float(r[mt])
                    long_rows.append(r)
                    meta[r['dataset']] = dict(n=r['n'], m_orig=r['m_original'], m=r['m_used'], d=r['d'])
                print(f'retomando: {len(meta)} conjunto(s) já concluído(s)')
    else:
        with open(long_csv, 'w', newline='') as fh:
            csv.DictWriter(fh, fieldnames=fields).writeheader()
    errors = []
    T0 = time.time()
    done_w = {}                                                    # pesos já calculados por conjunto
    for r in long_rows:
        done_w.setdefault(r['dataset'], set()).add(r['weight'])
    for ds in datasets:
        todo = [w for w in args.weights if w not in done_w.get(ds, set())]
        if ds in meta and 'gdt_full' in todo and int(meta[ds]['m']) > args.full_max_dim:
            todo.remove('gdt_full')                                # omitido por m grande, não por falta
        if not todo:
            print(f'[{ds}] já concluído — pulando'); continue
        if ds in meta:
            print(f'[{ds}] retomando só os pesos que faltam: {", ".join(todo)}')
        t0 = time.time()
        try:
            (X0, y0), src = load_dataset(ds, local, oml)
            for rep in range(args.repeats):
                X, y, m_orig, pinfo = preprocess(X0, y0, args.max_samples, args.pca_threshold,
                                                 args.pca_dims, args.seed + rep)
                if rep == 0:
                    print(f'[{ds}] {src}: n = {len(X)}, m = {m_orig}'
                          + (f' → {X.shape[1]} ({pinfo})' if pinfo else '') + f', classes = {len(np.unique(y))}')
                rows = run_one(X, y, args, todo, args.linkages, metrics)
                for r in rows:
                    r.update(dataset=ds, repeat=rep, source=src, n=len(X), m_original=m_orig,
                             m_used=X.shape[1], preproc=pinfo)
                with open(long_csv, 'a', newline='') as fh:
                    csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore').writerows(rows)
                long_rows += rows
            meta[ds] = dict(n=len(X), m_orig=m_orig, m=X.shape[1], d=rows[0]['d'])
            done_w.setdefault(ds, set()).update(todo)
            allr = [r for r in long_rows if r['dataset'] == ds and int(r['repeat']) == 0]
            best = {lk: max((r for r in allr if r['linkage'] == lk), key=lambda r: float(r['ARI']))
                    for lk in args.linkages if any(r['linkage'] == lk for r in allr)}
            print('      melhor por ligação (ARI): ' + '; '.join(
                f'{lk}: {b["weight"]} ({b["ARI"]:.3f})' for lk, b in best.items()) + f'  [{time.time() - t0:.0f}s]')
        except Exception as e:                                   # noqa: BLE001
            msg = f'{type(e).__name__}: {e}'
            print(f'[{ds}] FALHOU — {msg}')
            errors.append((ds, msg, traceback.format_exc()))
        write_tables(args.out, long_rows, meta, datasets, args.weights, args.linkages, metrics)  # parcial
    txt = write_tables(args.out, long_rows, meta, datasets, args.weights, args.linkages, metrics)
    if errors:
        with open(f'{args.out}_errors.txt', 'w') as fh:
            for ds, msg, tb in errors:
                fh.write(f'### {ds}\n{msg}\n{tb}\n')
    print('\n' + txt)
    tags = ', '.join(mt.replace('-', '') for mt in mnames)
    print(f'\narquivos: {args.out}_long.csv, {args.out}_<M>_wide.csv e {args.out}_<M>_table.tex (M = {tags}), '
          f'{args.out}_tables.tex, {args.out}_summary.txt'
          + (f', {args.out}_errors.txt ({len(errors)} falha(s): {", ".join(e[0] for e in errors)})' if errors else ''))
    print(f'tempo total: {(time.time() - T0) / 60:.1f} min')


if __name__ == '__main__':
    main()
