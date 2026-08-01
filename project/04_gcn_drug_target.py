"""
Stage 4: Semi-supervised GCN for Drug Target Identification

Replaces fixed-weight RWR with a learnable Graph Convolutional Network.
Uses known disease-gene associations as positive labels, non-seed proteins
as negative labels, and trains with weighted Binary Cross Entropy loss.

Comparison with RWR results from Stage 2 is built-in.

Usage:
    cd E:\zz\德国申请\LifeScience_projects\project
    python 04_gcn_drug_target.py

Output:
    results/gcn_ranked_targets_<disease>.csv   -- GCN ranking per disease
    results/gcn_vs_rwr_comparison.csv          -- side-by-side comparison
    results/plots/gcn_vs_rwr_top20.png         -- visualisation
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
PLOTS_DIR    = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────
HIDDEN_DIM   = 64       # GCN hidden layer size
DROPOUT      = 0.5      # dropout rate
LR           = 0.01     # learning rate
EPOCHS       = 300      # training epochs
SEED         = 42       # reproducibility

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── 1. Load data ─────────────────────────────────────────────────────────────

def safe_normalise(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return series * 0.0
    return (series - lo) / (hi - lo)


def load_data(disease: str):
    """
    Load PPI network and disease gene labels for one disease.
    Node features (7-dim):
      0: degree_norm          — local connectivity
      1: pagerank_norm        — global propagation influence
      2: betweenness_norm     — structural bottleneck position
      3: rwr_norm             — RWR score from Stage 2 (disease-specific)
      4: ot_score             — Open Targets association score (disease-specific)
      5: is_any_disease_gene  — 1 if seed gene in ANY of the 3 diseases
      6: degree_raw_norm      — raw degree (separate from centrality degree)
    """
    ppi_df     = pd.read_csv(DATA_DIR / "ppi_network.csv")
    disease_df = pd.read_csv(DATA_DIR / "disease_genes.csv")

    # Filter to one disease
    disease_genes = set(
        disease_df[disease_df["disease"] == disease]["symbol"].dropna()
    )

    # Build node index
    proteins = sorted(set(ppi_df["protein_a"]) | set(ppi_df["protein_b"]))
    node_idx  = {p: i for i, p in enumerate(proteins)}
    n_nodes   = len(proteins)

    # Edge index (undirected: add both directions)
    src = [node_idx[r.protein_a] for _, r in ppi_df.iterrows()]
    dst = [node_idx[r.protein_b] for _, r in ppi_df.iterrows()]
    edge_index = torch.tensor(
        [src + dst, dst + src], dtype=torch.long
    )

    # ── Load RWR results from Stage 2 (disease-specific) ──
    rwr_path = RESULTS_DIR / f"ranked_targets_{disease}.csv"
    rwr_df   = pd.read_csv(rwr_path) if rwr_path.exists() else pd.DataFrame()

    # ── Load Open Targets scores (disease-specific) ──
    ot_disease = disease_df[disease_df["disease"] == disease][["symbol", "ot_score"]]
    ot_dict    = dict(zip(ot_disease["symbol"], ot_disease["ot_score"]))

    # ── All-disease seed genes (feature 5) ──
    all_seeds = set(disease_df["symbol"].dropna())

    # ── Build feature matrix ──
    feat_dict = {}
    if not rwr_df.empty:
        for _, row in rwr_df.iterrows():
            feat_dict[row["protein"]] = {
                "degree_norm":     row.get("degree_norm", 0.0),
                "pagerank_norm":   row.get("pagerank_norm", 0.0),
                "betweenness_norm":row.get("betweenness_norm", 0.0),
                "rwr_norm":        row.get("rwr_norm", 0.0),
            }

    features = []
    for p in proteins:
        f = feat_dict.get(p, {})
        features.append([
            f.get("degree_norm",      0.0),
            f.get("pagerank_norm",    0.0),
            f.get("betweenness_norm", 0.0),
            f.get("rwr_norm",         0.0),
            ot_dict.get(p, 0.0),           # Open Targets score
            1.0 if p in all_seeds else 0.0, # any-disease seed flag
        ])

    x = torch.tensor(features, dtype=torch.float)   # [N, 6]

    # Normalise each feature column to [0,1]
    for col in range(x.shape[1]):
        col_min = x[:, col].min()
        col_max = x[:, col].max()
        if col_max > col_min:
            x[:, col] = (x[:, col] - col_min) / (col_max - col_min)

    # Labels: 1 = known disease gene (positive), 0 = non-seed (negative)
    # -1 = unlabelled (not used in loss) — used for hidden positive evaluation
    labels = torch.full((n_nodes,), -1, dtype=torch.float)

    pos_nodes = [node_idx[g] for g in disease_genes if g in node_idx]
    neg_nodes = [i for i, p in enumerate(proteins) if p not in disease_genes]

    # ── Missing Label Prediction setup ──
    # Hide 20% of positive labels → model must recover them from graph structure
    rng = np.random.default_rng(SEED)
    pos_shuffled  = rng.permutation(pos_nodes).tolist()
    n_hidden      = int(len(pos_shuffled) * 0.2)
    hidden_pos    = pos_shuffled[:n_hidden]   # hidden — label=-1, not in loss
    visible_pos   = pos_shuffled[n_hidden:]   # visible — label=1, in loss

    # Assign labels
    for i in visible_pos:
        labels[i] = 1.0
    for i in neg_nodes:
        labels[i] = 0.0
    # hidden_pos stays at -1 (unlabelled)

    # Train mask: visible positives + all negatives
    train_mask  = torch.zeros(n_nodes, dtype=torch.bool)
    hidden_mask = torch.zeros(n_nodes, dtype=torch.bool)

    for i in visible_pos + neg_nodes:
        train_mask[i] = True
    for i in hidden_pos:
        hidden_mask[i] = True

    # Positive weight
    pos_weight = torch.tensor([len(neg_nodes) / max(len(visible_pos), 1)], dtype=torch.float)

    print(f"  Disease: {disease}")
    print(f"  Nodes: {n_nodes}, Edges: {len(src)}")
    print(f"  Positive seeds — Visible (train): {len(visible_pos)}, Hidden (eval): {len(hidden_pos)}")
    print(f"  Negative nodes: {len(neg_nodes)}")
    print(f"  Pos weight: {pos_weight.item():.2f}")

    data = Data(x=x, edge_index=edge_index)
    return data, labels, train_mask, hidden_mask, hidden_pos, pos_weight, proteins, node_idx


# ── 2. GCN model ─────────────────────────────────────────────────────────────

class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1   = GCNConv(in_dim, hidden_dim)
        self.conv2   = GCNConv(hidden_dim, hidden_dim)
        self.out     = torch.nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.out(x).squeeze(-1)   # [N]
        return x


# ── 3. Train ──────────────────────────────────────────────────────────────────

def train(disease: str):
    print(f"\n{'='*60}")
    print(f"  Training GCN — {disease}")
    print(f"{'='*60}")

    data, labels, train_mask, hidden_mask, hidden_pos, pos_weight, proteins, node_idx = load_data(disease)

    print(f"  Node feature dimensions: {data.x.shape[1]}")
    model     = GCN(in_dim=data.x.shape[1], hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    loss_fn   = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss   = loss_fn(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:>3} | Train Loss: {loss.item():.4f}")

    # ── Missing Label Evaluation ──
    model.eval()
    with torch.no_grad():
        logits = model(data)
        scores = torch.sigmoid(logits).numpy()

    # Rank all nodes by GCN score
    all_scores = list(enumerate(scores))
    all_scores.sort(key=lambda x: x[1], reverse=True)
    rank_of = {idx: rank+1 for rank, (idx, _) in enumerate(all_scores)}

    # Ranks of hidden positive nodes
    hidden_ranks = sorted([rank_of[i] for i in hidden_pos])
    n_hidden     = len(hidden_pos)
    n_total      = len(proteins)

    # How many hidden positives appear in top-K
    for k in [100, 200, 500]:
        hits = sum(1 for r in hidden_ranks if r <= k)
        pct  = hits / n_hidden * 100
        print(f"  Hidden positives in top-{k:<4}: {hits}/{n_hidden} ({pct:.1f}%)")

    median_rank = np.median(hidden_ranks)
    mean_rank   = np.mean(hidden_ranks)
    print(f"  Hidden positives — Median rank: {median_rank:.0f} / {n_total} "
          f"| Mean rank: {mean_rank:.0f} / {n_total}")
    print(f"  (Random baseline median rank would be ~{n_total//2})")
    print(f"  {'─'*60}")

    # ── Inference on all nodes ──
    model.eval()
    with torch.no_grad():
        logits = model(data)
        scores = torch.sigmoid(logits).numpy()

    # Build result DataFrame
    disease_genes = set(
        pd.read_csv(DATA_DIR / "disease_genes.csv")
        .query(f"disease == '{disease}'")["symbol"].dropna()
    )

    records = []
    for i, protein in enumerate(proteins):
        records.append({
            "protein":      protein,
            "gcn_score":    round(float(scores[i]), 5),
            "is_seed_gene": protein in disease_genes,
        })

    df = (
        pd.DataFrame(records)
        .sort_values("gcn_score", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "gcn_rank", df.index + 1)

    # Save
    out_path = RESULTS_DIR / f"gcn_ranked_targets_{disease}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    return df, proteins


# ── 4. Compare GCN vs RWR ────────────────────────────────────────────────────

def compare_gcn_rwr(disease: str, gcn_df: pd.DataFrame):
    """Load existing RWR results and compare with GCN rankings."""
    rwr_path = RESULTS_DIR / f"ranked_targets_{disease}.csv"
    if not rwr_path.exists():
        print(f"  RWR results not found for {disease}, skipping comparison.")
        return None

    rwr_df = pd.read_csv(rwr_path)[["protein", "rank", "composite_score", "is_seed_gene"]]
    rwr_df = rwr_df.rename(columns={"rank": "rwr_rank", "composite_score": "rwr_score"})

    merged = gcn_df[["protein", "gcn_rank", "gcn_score"]].merge(
        rwr_df, on="protein", how="inner"
    )

    # Non-seed only (novel candidates)
    novel = merged[~merged["is_seed_gene"]].copy()

    print(f"\n  {'='*50}")
    print(f"  GCN vs RWR — {disease} — Top 20 non-seed candidates")
    print(f"  {'='*50}")
    print(f"  {'Protein':<12} {'GCN Rank':>9} {'RWR Rank':>9} {'Δ Rank':>8}")
    print(f"  {'-'*42}")
    for _, row in novel.head(20).iterrows():
        delta = int(row["rwr_rank"]) - int(row["gcn_rank"])
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {row['protein']:<12} {int(row['gcn_rank']):>9} "
              f"{int(row['rwr_rank']):>9} {arrow}{abs(delta):>6}")

    # Known targets recovery
    known_targets = {"TNF", "IL6", "JAK2", "IL17A", "VEGFA"}
    print(f"\n  Known target recovery (GCN vs RWR):")
    for target in known_targets:
        row = merged[merged["protein"] == target]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {target:<10} GCN #{int(r['gcn_rank']):<6} RWR #{int(r['rwr_rank']):<6}")

    return merged


# ── 5. Visualise ─────────────────────────────────────────────────────────────

def plot_comparison(merged_df: pd.DataFrame, disease: str):
    """Plot GCN vs RWR top-20 non-seed candidates side by side."""
    novel = merged_df[~merged_df["is_seed_gene"]].copy()
    top20 = novel.head(20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"GCN vs RWR — Top 20 Novel Candidates ({disease})", fontsize=13)

    for ax, score_col, rank_col, label, color in [
        (axes[0], "gcn_score",  "gcn_rank",  "GCN Score",  "#2c3e7a"),
        (axes[1], "rwr_score",  "rwr_rank",  "RWR Composite Score", "steelblue"),
    ]:
        # Sort by this method's rank
        plot_df = top20.sort_values(rank_col)
        ax.barh(plot_df["protein"][::-1], plot_df[score_col][::-1],
                color=color, alpha=0.85, edgecolor="white")
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(label.split()[0], fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = PLOTS_DIR / f"gcn_vs_rwr_{disease}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved plot → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    diseases = ["multiple_sclerosis", "rheumatoid_arthritis", "systemic_lupus"]

    all_comparisons = []
    for disease in diseases:
        gcn_df, proteins = train(disease)
        merged = compare_gcn_rwr(disease, gcn_df)
        if merged is not None:
            merged["disease"] = disease
            all_comparisons.append(merged)
            plot_comparison(merged, disease)

    if all_comparisons:
        combined = pd.concat(all_comparisons, ignore_index=True)
        out = RESULTS_DIR / "gcn_vs_rwr_comparison.csv"
        combined.to_csv(out, index=False)
        print(f"\nFull comparison saved → {out}")

    print(f"\n{'='*60}")
    print("  Done. Check results/ for CSVs and results/plots/ for figures.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
