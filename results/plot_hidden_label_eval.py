"""
plot_hidden_label_eval.py
Missing Label Prediction可视化
用GCN对隐藏的20% seed genes做排名，对比随机基线
输出：results/plots/hidden_label_evaluation.png
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'project'))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

RESULTS_DIR = Path(__file__).parent
PLOTS_DIR   = RESULTS_DIR / "plots"
DATA_DIR    = RESULTS_DIR.parent / "data"
PLOTS_DIR.mkdir(exist_ok=True)

SEED       = 42
HIDDEN_DIM = 64
DROPOUT    = 0.5
LR         = 0.01
EPOCHS     = 150
HIDE_RATIO = 0.2

np.random.seed(SEED)
torch.manual_seed(SEED)

DISEASES = {
    "multiple_sclerosis":   "MS",
    "rheumatoid_arthritis": "RA",
    "systemic_lupus":       "SLE",
}


class GCN(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, HIDDEN_DIM)
        self.out   = torch.nn.Linear(HIDDEN_DIM, 1)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index))
        return self.out(x).squeeze(-1)


def run_disease(disease: str):
    ppi_df     = pd.read_csv(DATA_DIR / "ppi_network.csv")
    disease_df = pd.read_csv(DATA_DIR / "disease_genes.csv")
    rwr_df     = pd.read_csv(RESULTS_DIR / f"ranked_targets_{disease}.csv")

    disease_genes = set(disease_df[disease_df["disease"] == disease]["symbol"].dropna())
    ot            = dict(zip(disease_df[disease_df["disease"]==disease]["symbol"],
                             disease_df[disease_df["disease"]==disease]["ot_score"]))
    all_seeds     = set(disease_df["symbol"].dropna())

    proteins  = sorted(set(ppi_df["protein_a"]) | set(ppi_df["protein_b"]))
    node_idx  = {p: i for i, p in enumerate(proteins)}
    n_nodes   = len(proteins)

    src = [node_idx[r.protein_a] for _, r in ppi_df.iterrows()]
    dst = [node_idx[r.protein_b] for _, r in ppi_df.iterrows()]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    feat_dict = {r["protein"]: r for _, r in rwr_df.iterrows()}
    features  = []
    for p in proteins:
        f = feat_dict.get(p, {})
        features.append([
            float(f.get("degree_norm", 0)),
            float(f.get("pagerank_norm", 0)),
            float(f.get("betweenness_norm", 0)),
            float(f.get("rwr_norm", 0)),
            ot.get(p, 0.0),
            1.0 if p in all_seeds else 0.0,
        ])
    x = torch.tensor(features, dtype=torch.float)
    for col in range(x.shape[1]):
        lo, hi = x[:, col].min(), x[:, col].max()
        if hi > lo:
            x[:, col] = (x[:, col] - lo) / (hi - lo)

    pos_nodes = [node_idx[g] for g in disease_genes if g in node_idx]
    neg_nodes = [i for i, p in enumerate(proteins) if p not in disease_genes]

    rng          = np.random.default_rng(SEED)
    pos_shuffled = rng.permutation(pos_nodes).tolist()
    n_hidden     = int(len(pos_shuffled) * HIDE_RATIO)
    hidden_pos   = pos_shuffled[:n_hidden]
    visible_pos  = pos_shuffled[n_hidden:]

    labels = torch.full((n_nodes,), -1, dtype=torch.float)
    for i in visible_pos: labels[i] = 1.0
    for i in neg_nodes:   labels[i] = 0.0

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    for i in visible_pos + neg_nodes:
        train_mask[i] = True

    pos_weight = torch.tensor([len(neg_nodes) / max(len(visible_pos), 1)])
    loss_fn    = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    data       = Data(x=x, edge_index=edge_index)

    model     = GCN(in_dim=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(data)[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(data)).numpy()

    ranked = sorted(range(n_nodes), key=lambda i: scores[i], reverse=True)
    rank_of = {idx: rank + 1 for rank, idx in enumerate(ranked)}
    hidden_ranks = sorted([rank_of[i] for i in hidden_pos])

    return hidden_ranks, n_nodes, n_hidden


# ── 收集三种疾病的数据 ────────────────────────────────────────────────────────
all_results = {}
for disease, label in DISEASES.items():
    print(f"Running {label}...")
    ranks, n_total, n_hidden = run_disease(disease)
    all_results[label] = {"ranks": ranks, "n_total": n_total, "n_hidden": n_hidden}
    print(f"  Median rank: {np.median(ranks):.0f} / {n_total}")

# ── 可视化 ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Missing Label Prediction: GCN Recovers Hidden Seed Genes\n"
             "(20% of seed genes hidden from training — model must rediscover them from graph structure)",
             fontsize=12, fontweight='bold')

# ── 左图：Cumulative recovery curve ──────────────────────────────────────────
ax = axes[0]
colors = {"MS": "#2c3e7a", "RA": "#e67e22", "SLE": "#27ae60"}

for label, res in all_results.items():
    ranks   = res["ranks"]
    n_total = res["n_total"]
    n_hid   = res["n_hidden"]

    ks     = list(range(1, n_total + 1))
    recall = [sum(1 for r in ranks if r <= k) / n_hid for k in ks]
    ax.plot(ks, recall, color=colors[label], linewidth=2, label=f"GCN — {label}")

# Random baseline
random_recall = [k / list(all_results.values())[0]["n_total"] for k in
                 range(1, list(all_results.values())[0]["n_total"] + 1)]
ax.plot(range(1, list(all_results.values())[0]["n_total"] + 1),
        random_recall, 'k--', linewidth=1, alpha=0.4, label="Random baseline")

ax.axvline(x=100,  color='grey', linestyle=':', alpha=0.6, linewidth=0.8)
ax.axvline(x=500,  color='grey', linestyle=':', alpha=0.6, linewidth=0.8)
ax.text(100,  0.02, 'Top-100',  fontsize=7, color='grey')
ax.text(500,  0.02, 'Top-500',  fontsize=7, color='grey')
ax.set_xlim(0, 1000)
ax.set_xlabel("Top-K candidates", fontsize=10)
ax.set_ylabel("Recall (fraction of hidden positives recovered)", fontsize=10)
ax.set_title("Cumulative Recovery Curve", fontsize=11)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── 右图：Bar chart top-K hits across diseases ────────────────────────────────
ax2 = axes[1]
ks     = [100, 200, 500]
x      = np.arange(len(ks))
width  = 0.25

for i, (label, res) in enumerate(all_results.items()):
    ranks  = res["ranks"]
    n_hid  = res["n_hidden"]
    pcts   = [sum(1 for r in ranks if r <= k) / n_hid * 100 for k in ks]
    bars   = ax2.bar(x + i * width, pcts, width,
                     label=label, color=colors[label], alpha=0.85, edgecolor='white')
    for bar, pct in zip(bars, pcts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{pct:.0f}%', ha='center', fontsize=8,
                 color=colors[label], fontweight='bold')

# Random baseline bars
random_pcts = [k / list(all_results.values())[0]["n_total"] * 100 for k in ks]
for j, (k, rp) in enumerate(zip(ks, random_pcts)):
    ax2.axhline(y=rp, xmin=(j * 1/3) + 0.02, xmax=((j+1) * 1/3) - 0.02,
                color='black', linestyle='--', alpha=0.4, linewidth=1.5)

ax2.set_xticks(x + width)
ax2.set_xticklabels([f"Top-{k}" for k in ks], fontsize=10)
ax2.set_ylabel("Hidden positives recovered (%)", fontsize=10)
ax2.set_title("Recovery Rate at Top-K\n(dashed = random baseline)", fontsize=11)
ax2.legend(fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(0, 115)

plt.tight_layout()
path = PLOTS_DIR / "hidden_label_evaluation.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved → {path}")
