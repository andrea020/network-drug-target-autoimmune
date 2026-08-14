"""
plot_comparison.py
生成RWR vs GCN对比图，用于Technical Report
输出：results/plots/rwr_vs_gcn_report.png
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(RESULTS_DIR / "gcn_vs_rwr_comparison.csv")

# ── 图1：三种疾病的已知靶点覆盖率对比（Precision@K）────────────────────────
DISEASE_LABELS = {
    "multiple_sclerosis":   "MS",
    "rheumatoid_arthritis": "RA",
    "systemic_lupus":       "SLE",
}
diseases  = list(DISEASE_LABELS.keys())
TOP_K     = 100  # Precision@100

disease_df = pd.read_csv(Path(__file__).parent.parent / "data" / "disease_genes.csv")

rwr_precision = []
gcn_precision = []
total_known   = []

for disease in diseases:
    d = df[df['disease'] == disease].copy()

    # Ground truth：该疾病的seed genes（Open Targets score >= 0.15）
    known = set(disease_df[disease_df['disease'] == disease]['symbol'].dropna())
    total_known.append(len(known))

    # 非种子蛋白里，RWR top-K和GCN top-K各覆盖多少已知靶点
    # 注意：non-seed里不包含seed genes，所以这里用所有蛋白（含seed）算覆盖率
    # 因为评估的是方法能不能把已知靶点排到前面
    rwr_top_k = set(d.nsmallest(TOP_K, 'rwr_rank')['protein'])
    gcn_top_k = set(d.nsmallest(TOP_K, 'gcn_rank')['protein'])

    rwr_hits = len(known & rwr_top_k)
    gcn_hits = len(known & gcn_top_k)

    rwr_precision.append(rwr_hits / TOP_K * 100)
    gcn_precision.append(gcn_hits / TOP_K * 100)

fig, ax = plt.subplots(figsize=(8, 5))

x     = np.arange(len(diseases))
width = 0.35

bars1 = ax.bar(x - width/2, rwr_precision, width, label='RWR',
               color='#2c3e7a', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, gcn_precision, width, label='GCN (semi-supervised)',
               color='#e67e22', alpha=0.85, edgecolor='white')

# 数值标注
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=9, color='#2c3e7a', fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=9, color='#e67e22', fontweight='bold')

labels_with_n = [f'{DISEASE_LABELS[d]}\n(n={total_known[i]} known targets)'
                 for i, d in enumerate(diseases)]
ax.set_xticks(x)
ax.set_xticklabels(labels_with_n, fontsize=10)
ax.set_ylabel('Known Target Coverage in Top-100 (%)', fontsize=10)
ax.set_title(f'Precision@{TOP_K}: Known Target Recovery\n'
             f'(Ground truth: Open Targets disease-gene associations, score ≥ 0.15)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(rwr_precision + gcn_precision) * 1.3)

plt.tight_layout()
path1 = PLOTS_DIR / "rwr_vs_gcn_precision_at_k.png"
plt.savefig(path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {path1}")


# ── 图2：MS的Rank变化散点图（RWR rank vs GCN rank，高亮FAU和已知靶点）────────
fig, ax = plt.subplots(figsize=(8, 7))

ms = df[df['disease'] == 'multiple_sclerosis']
novel_ms = ms[~ms['is_seed_gene']].copy()

# 只画rank < 500的，避免太密集
plot_df = novel_ms[(novel_ms['rwr_rank'] < 500) | (novel_ms['gcn_rank'] < 500)].copy()

# 普通点
ax.scatter(plot_df['rwr_rank'], plot_df['gcn_rank'],
           alpha=0.3, s=20, color='#95a5a6', label='Other proteins')

# JAK家族高亮
jak_proteins = ['JAK1', 'JAK3', 'TYK2', 'JAK2']
jak_df = novel_ms[novel_ms['protein'].isin(jak_proteins)]
ax.scatter(jak_df['rwr_rank'], jak_df['gcn_rank'],
           s=80, color='#27ae60', zorder=5, label='JAK family (MS-relevant)')
for _, row in jak_df.iterrows():
    ax.annotate(row['protein'],
                (row['rwr_rank'], row['gcn_rank']),
                xytext=(5, 5), textcoords='offset points', fontsize=8, color='#27ae60')

# FAU高亮
fau = novel_ms[novel_ms['protein'] == 'FAU']
if not fau.empty:
    ax.scatter(fau['rwr_rank'], fau['gcn_rank'],
               s=150, color='#e74c3c', zorder=6,
               marker='*', label='FAU (novel MS candidate)')
    ax.annotate('FAU',
                (fau['rwr_rank'].iloc[0], fau['gcn_rank'].iloc[0]),
                xytext=(8, -12), textcoords='offset points',
                fontsize=10, color='#e74c3c', fontweight='bold')

# 对角线：两种方法排名相同
max_rank = 500
ax.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.2, linewidth=0.8, label='Equal rank')

ax.set_xlabel('RWR Rank', fontsize=11)
ax.set_ylabel('GCN Rank', fontsize=11)
ax.set_title('MS: RWR vs GCN Rank Comparison (Non-seed Proteins)\n'
             'Points above diagonal = GCN ranks lower than RWR',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, max_rank)
ax.set_ylim(0, max_rank)

plt.tight_layout()
path2 = PLOTS_DIR / "rwr_vs_gcn_ms_scatter.png"
plt.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {path2}")


# ── 图3：FAU在三种疾病里的排名对比 ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

diseases = ['multiple_sclerosis', 'rheumatoid_arthritis', 'systemic_lupus']
labels   = ['MS', 'RA', 'SLE']
fau_rwr  = []
fau_gcn  = []

for disease in diseases:
    d = df[df['disease'] == disease]
    row = d[d['protein'] == 'FAU']
    if not row.empty:
        fau_rwr.append(int(row['rwr_rank'].iloc[0]))
        fau_gcn.append(int(row['gcn_rank'].iloc[0]))
    else:
        fau_rwr.append(None)
        fau_gcn.append(None)

x = np.arange(len(labels))
width = 0.35

ax.bar(x - width/2, fau_rwr, width, label='RWR Rank', color='#2c3e7a', alpha=0.85)
ax.bar(x + width/2, fau_gcn, width, label='GCN Rank', color='#e67e22', alpha=0.85)

for i, (r, g) in enumerate(zip(fau_rwr, fau_gcn)):
    if r: ax.text(i - width/2, r + 5, f'#{r}', ha='center', fontsize=9, color='#2c3e7a', fontweight='bold')
    if g: ax.text(i + width/2, g + 5, f'#{g}', ha='center', fontsize=9, color='#e67e22', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Rank (lower = better)', fontsize=10)
ax.set_title('FAU Ranking: RWR vs GCN Across Three Autoimmune Diseases\n'
             '(FAU has no curated autoimmune annotation in DisGeNET)',
             fontsize=11, fontweight='bold')
ax.invert_yaxis()
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
path3 = PLOTS_DIR / "fau_rwr_vs_gcn.png"
plt.savefig(path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {path3}")

print("\n=== All plots saved ===")
print(f"1. Known target recovery: {path1}")
print(f"2. MS rank scatter:       {path2}")
print(f"3. FAU ranking:           {path3}")
