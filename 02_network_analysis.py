"""
Stage 2: Build PPI network and rank drug target candidates

Methods used:
  1. NetworkX graph construction from STRING interactions
  2. Classical centrality metrics (degree, PageRank, betweenness)
  3. Random Walk with Restart (RWR) — network propagation from disease seed genes
  4. Composite scoring + per-disease ranking

Usage:
    python 02_network_analysis.py

Output:
    results/ranked_targets.csv    -- all proteins ranked by composite score
    results/plots/                -- centrality distributions + disease module figure
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe on all platforms)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

PPI_MIN_SCORE = 0.7   # keep only high-confidence edges (STRING score ≥ 0.7)
RWR_ALPHA = 0.85      # restart probability for Random Walk with Restart
RWR_ITERATIONS = 100


# ---------------------------------------------------------------------------
# 1.  Graph construction
# ---------------------------------------------------------------------------

def build_graph(ppi_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    filtered = ppi_df[ppi_df["score"] >= PPI_MIN_SCORE]
    for _, row in filtered.iterrows():
        G.add_edge(row["protein_a"], row["protein_b"], weight=float(row["score"]))
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(score ≥ {PPI_MIN_SCORE})")
    return G


# ---------------------------------------------------------------------------
# 2.  Network propagation — Random Walk with Restart
# ---------------------------------------------------------------------------

def random_walk_with_restart(G: nx.Graph, seed_genes: list[str]) -> dict[str, float]:
    """
    Propagate signal from seed genes through the network.
    Returns a score for every node: higher = closer to disease genes.
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}

    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Row-normalised adjacency matrix (weights = STRING scores)
    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = (A / row_sums).T  # column-stochastic transition matrix

    # Seed vector  p0
    seed_set = set(seed_genes) & set(nodes)
    if not seed_set:
        return {n: 0.0 for n in nodes}

    p0 = np.zeros(n)
    for gene in seed_set:
        p0[node_idx[gene]] = 1.0
    p0 /= p0.sum()

    # Iterate: p_t+1 = alpha * W @ p_t + (1 - alpha) * p0
    p = p0.copy()
    for _ in range(RWR_ITERATIONS):
        p_new = RWR_ALPHA * W @ p + (1.0 - RWR_ALPHA) * p0
        if np.linalg.norm(p_new - p, ord=1) < 1e-8:
            break
        p = p_new

    return {nodes[i]: float(p[i]) for i in range(n)}


# ---------------------------------------------------------------------------
# 3.  Centrality metrics + composite score
# ---------------------------------------------------------------------------

def compute_centralities(G: nx.Graph) -> dict[str, dict]:
    print("Computing degree centrality ...")
    degree_c = nx.degree_centrality(G)

    print("Computing PageRank ...")
    pagerank = nx.pagerank(G, weight="weight", alpha=0.85)

    print("Computing betweenness centrality (approximated for large graphs) ...")
    k = min(300, G.number_of_nodes())
    betweenness = nx.betweenness_centrality(G, k=k, weight="weight", normalized=True)

    return {"degree": degree_c, "pagerank": pagerank, "betweenness": betweenness}


def _safe_normalise(scores: dict) -> dict:
    """Min-max normalise a score dict to [0, 1]."""
    vals = list(scores.values())
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def rank_targets(G: nx.Graph, disease_df: pd.DataFrame, centralities: dict,
                 disease_filter: str = None) -> pd.DataFrame:
    if disease_filter:
        subset = disease_df[disease_df["disease"] == disease_filter]
    else:
        subset = disease_df
    seed_genes = subset["symbol"].dropna().unique().tolist()
    print(f"\nRunning RWR with {len(seed_genes)} seed genes ...")
    rwr_raw = random_walk_with_restart(G, seed_genes)

    # Normalise all metrics to [0, 1] before combining
    deg_n  = _safe_normalise(centralities["degree"])
    pr_n   = _safe_normalise(centralities["pagerank"])
    bw_n   = _safe_normalise(centralities["betweenness"])
    rwr_n  = _safe_normalise(rwr_raw) if rwr_raw else {n: 0.0 for n in G.nodes()}

    seed_set = set(seed_genes)
    records = []
    for node in G.nodes():
        # Composite: RWR and PageRank carry most weight; betweenness rewards bottlenecks
        composite = (
            0.35 * rwr_n.get(node, 0)
            + 0.30 * pr_n.get(node, 0)
            + 0.20 * bw_n.get(node, 0)
            + 0.15 * deg_n.get(node, 0)
        )
        records.append({
            "protein":         node,
            "degree":          G.degree(node),
            "degree_norm":     round(deg_n.get(node, 0), 5),
            "pagerank_norm":   round(pr_n.get(node, 0), 5),
            "betweenness_norm": round(bw_n.get(node, 0), 5),
            "rwr_norm":        round(rwr_n.get(node, 0), 5),
            "composite_score": round(composite, 5),
            "is_seed_gene":    node in seed_set,
        })

    df = (
        pd.DataFrame(records)
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "rank", df.index + 1)
    return df


# ---------------------------------------------------------------------------
# 4.  Visualisations
# ---------------------------------------------------------------------------

def plot_score_distributions(ranked_df: pd.DataFrame):
    metrics = ["degree_norm", "pagerank_norm", "betweenness_norm", "rwr_norm"]
    labels  = ["Degree", "PageRank", "Betweenness", "RWR Score"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Normalised Centrality Distributions (PPI Network)", fontsize=13)

    for ax, col, lbl in zip(axes, metrics, labels):
        ax.hist(ranked_df[col], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(lbl)
        ax.set_xlabel("Score (0–1)")
        ax.set_ylabel("# Proteins")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = PLOTS_DIR / "centrality_distributions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_top_targets(ranked_df: pd.DataFrame, top_n: int = 25):
    top = ranked_df.head(top_n).copy()
    colors = ["#d62728" if s else "#1f77b4" for s in top["is_seed_gene"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top["protein"][::-1], top["composite_score"][::-1],
                   color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Composite Score", fontsize=11)
    ax.set_title(f"Top {top_n} Drug Target Candidates\n"
                 f"(red = known autoimmune disease gene)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#d62728", label="Known disease gene (seed)"),
        Patch(color="#1f77b4", label="Novel candidate"),
    ], loc="lower right", fontsize=9)

    plt.tight_layout()
    path = PLOTS_DIR / "top_targets.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_disease_module_subgraph(G: nx.Graph, ranked_df: pd.DataFrame,
                                 disease_df: pd.DataFrame, top_n: int = 30):
    top_proteins = set(ranked_df.head(top_n)["protein"])
    seed_set = set(disease_df["symbol"].dropna())

    subgraph = G.subgraph(top_proteins)
    if subgraph.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(subgraph, seed=42, k=1.5)
    node_colors = ["#d62728" if n in seed_set else "#4c9be8" for n in subgraph.nodes()]
    node_sizes  = [300 + 800 * ranked_df.set_index("protein")["composite_score"].get(n, 0)
                   for n in subgraph.nodes()]

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(subgraph, pos, alpha=0.25, edge_color="grey", ax=ax)
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(subgraph, pos, font_size=7, ax=ax)
    ax.set_title(f"Disease Module Subgraph — Top {top_n} Candidates", fontsize=13)
    ax.axis("off")

    path = PLOTS_DIR / "disease_module_subgraph.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 5.  Console summary
# ---------------------------------------------------------------------------

def print_top_targets(ranked_df: pd.DataFrame, top_n: int = 20):
    print(f"\n{'='*65}")
    print(f"  TOP {top_n} DRUG TARGET CANDIDATES")
    print(f"{'='*65}")
    header = f"{'Rank':<5} {'Protein':<12} {'Degree':<8} {'Composite':>10}  Note"
    print(header)
    print("-" * 65)
    for _, row in ranked_df.head(top_n).iterrows():
        note = "(known seed)" if row["is_seed_gene"] else ""
        print(f"{row['rank']:<5} {row['protein']:<12} {row['degree']:<8} "
              f"{row['composite_score']:>10.4f}  {note}")


def print_novel_candidates(ranked_df: pd.DataFrame, disease_df: pd.DataFrame,
                           top_n: int = 10, label: str = "combined"):
    seed_set = set(disease_df["symbol"].dropna())
    novel = ranked_df[~ranked_df["protein"].isin(seed_set)].head(top_n)
    print(f"\n{'='*65}")
    print(f"  TOP {top_n} NOVEL CANDIDATES — {label}")
    print(f"{'='*65}")
    for _, row in novel.iterrows():
        print(f"  #{row['rank']:>3}  {row['protein']:<12}  composite={row['composite_score']:.4f}  "
              f"degree={row['degree']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ppi_path     = DATA_DIR / "ppi_network.csv"
    disease_path = DATA_DIR / "disease_genes.csv"

    if not ppi_path.exists() or not disease_path.exists():
        raise FileNotFoundError(
            "Missing input data. Run 01_fetch_data.py first."
        )

    ppi_df     = pd.read_csv(ppi_path)
    disease_df = pd.read_csv(disease_path)
    print(f"Loaded: {len(ppi_df)} interactions, {len(disease_df)} disease-gene rows\n")

    G = build_graph(ppi_df)

    if G.number_of_nodes() == 0:
        print("\nERROR: PPI network is empty — cannot run analysis.")
        print("Diagnostic info:")
        print(f"  ppi_network.csv rows : {len(ppi_df)}")
        if len(ppi_df) > 0:
            print(f"  score range          : {ppi_df['score'].min():.3f} – {ppi_df['score'].max():.3f}")
            print(f"  rows with score≥{PPI_MIN_SCORE}  : {(ppi_df['score'] >= PPI_MIN_SCORE).sum()}")
            print(f"  → Try lowering PPI_MIN_SCORE from {PPI_MIN_SCORE} to 0.4 at the top of this file")
        else:
            print("  → ppi_network.csv is empty. Re-run 01_fetch_data.py")
        return

    centralities = compute_centralities(G)

    # Combined run (all diseases as seeds)
    ranked_df = rank_targets(G, disease_df, centralities)
    out = RESULTS_DIR / "ranked_targets_combined.csv"
    ranked_df.to_csv(out, index=False)
    print(f"\nRanked targets saved → {out}")

    print("\nGenerating plots ...")
    plot_score_distributions(ranked_df)
    plot_top_targets(ranked_df)
    plot_disease_module_subgraph(G, ranked_df, disease_df)

    print_top_targets(ranked_df)
    print_novel_candidates(ranked_df, disease_df, label="combined")

    # Per-disease runs
    for disease_key in sorted(disease_df["disease"].unique()):
        ranked_d = rank_targets(G, disease_df, centralities, disease_filter=disease_key)
        out_d = RESULTS_DIR / f"ranked_targets_{disease_key}.csv"
        ranked_d.to_csv(out_d, index=False)
        d_subset = disease_df[disease_df["disease"] == disease_key]
        print_novel_candidates(ranked_d, d_subset, label=disease_key)

    print(f"\n{'='*65}")
    print("  Done! Check results/ for CSV and results/plots/ for figures.")
    print("="*65)


if __name__ == "__main__":
    main()
