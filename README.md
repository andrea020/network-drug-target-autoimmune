# Network-Based Drug Target Identification for Autoimmune Diseases

A computational pipeline that applies graph propagation to prioritise drug target candidates in three autoimmune diseases: **Rheumatoid Arthritis (RA)**, **Systemic Lupus Erythematosus (SLE)**, and **Multiple Sclerosis (MS)**.

The pipeline recovers established clinical targets (TNF, IL6, JAK2) in the top 10 and identifies **FAU** — a protein with no prior autoimmune disease association in DisGeNET, whose disease profile is otherwise dominated by oncology annotations — as the top-ranked non-seed candidate in MS, robust across six scoring configurations.

---

## Pipeline Overview

```
Open Targets API          STRING DB API
(disease-gene scores)     (PPI interactions)
        │                        │
        └──────────┬─────────────┘
                   ▼
          01_fetch_data.py
          (seed genes + PPI network)
                   │
                   ▼
        02_network_analysis.py
        (graph construction, RWR,
         centrality metrics, ranking)
                   │
                   ▼
         03_validate_fau.py
         (literature, DisGeNET,
          robustness analysis)
```

---

## Methods

### Stage 1 — Data Acquisition (`01_fetch_data.py`)

- **Disease seed genes**: queried from [Open Targets](https://platform.opentargets.org/) GraphQL API (association score ≥ 0.15; up to 300 genes per disease)
- **PPI network**: queried from [STRING DB](https://string-db.org/) REST API (combined score ≥ 700; top 25 interaction partners per seed gene)
- Data cached locally to `data/` to avoid re-fetching

### Stage 2 — Network Analysis (`02_network_analysis.py`)

**Graph construction**: NetworkX undirected graph from STRING interactions (edge weight = STRING combined score, filtered at ≥ 0.7)

**Centrality metrics** (three independent dimensions):
| Metric | What it measures |
|--------|-----------------|
| Degree centrality | Local connectivity (number of direct neighbours) |
| PageRank | Global propagation influence (iterative, weighted) |
| Betweenness centrality | Structural bottleneck position (bridge nodes) |

**Random Walk with Restart (RWR)**: network propagation from disease seed genes with restart probability α = 0.85, iterated to convergence (tolerance 1e-8). Score reflects proximity to the disease gene neighbourhood.

**Composite scoring** (all metrics normalised to [0, 1] before combining):

```
composite = 0.35 × RWR + 0.30 × PageRank + 0.20 × Betweenness + 0.15 × Degree
```

Separate rankings are produced for each disease and a combined (all-disease) run.

### Stage 3 — FAU Validation (`03_validate_fau.py`)

Three independent validation approaches for the top non-seed candidate (FAU, ranked #2 in MS):

**A. Literature search** — Semantic Scholar queries for FAU in autoimmune/immune contexts  
**B. Database lookup** — DisGeNET REST API for all disease associations of FAU (NCBI Gene ID: 2197)  
**C. Robustness analysis** — FAU rank tracked across six composite weight configurations:

| Configuration | RWR | PageRank | Betweenness | Degree |
|--------------|-----|----------|-------------|--------|
| Current | 0.35 | 0.30 | 0.20 | 0.15 |
| RWR dominant | 0.60 | 0.20 | 0.10 | 0.10 |
| PageRank dominant | 0.20 | 0.50 | 0.20 | 0.10 |
| Equal weights | 0.25 | 0.25 | 0.25 | 0.25 |
| RWR only | 1.00 | 0.00 | 0.00 | 0.00 |
| Topology only | 0.00 | 0.40 | 0.35 | 0.25 |

---

## Key Result

**FAU** (FUBI-RPS30, NCBI Gene ID: 2197) ranks #2 among all non-seed proteins in the MS analysis (composite score: 0.7004, RWR normalised: 1.0). Its disease profile in DisGeNET is dominated by oncology annotations with **no autoimmune disease associations**, making it a novel computational prediction for MS drug target investigation.

Established clinical targets (TNF, IL6, JAK2) are recovered in the top 10 across all disease analyses, confirming pipeline validity before examining novel candidates.

---

## Repository Structure

```
├── project/
│   ├── 01_fetch_data.py          # Data acquisition (Open Targets + STRING)
│   ├── 02_network_analysis.py    # Graph construction, RWR, ranking
│   ├── 03_validate_fau.py        # FAU three-pronged validation
│   └── requirements.txt
├── data/
│   ├── disease_genes.csv         # Seed genes per disease (auto-generated)
│   └── ppi_network.csv           # PPI edges with STRING scores (auto-generated)
├── results/
│   ├── ranked_targets_combined.csv
│   ├── ranked_targets_multiple_sclerosis.csv
│   ├── ranked_targets_rheumatoid_arthritis.csv
│   ├── ranked_targets_systemic_lupus.csv
│   ├── plots/
│   │   ├── centrality_distributions.png
│   │   ├── top_targets.png
│   │   └── disease_module_subgraph.png
│   └── fau_validation/
│       ├── literature_fau.md
│       ├── disgenet_fau.csv
│       ├── robustness_fau.csv
│       └── robustness_fau.png
└── README.md
```

---

## Quickstart

```bash
cd project/
pip install -r requirements.txt

python 01_fetch_data.py        # ~2 min; fetches seed genes + PPI network
python 02_network_analysis.py  # ~3 min; builds graph, runs RWR, produces rankings
python 03_validate_fau.py      # ~5 min; validates FAU (Semantic Scholar has rate limits)
```

**Note**: `01_fetch_data.py` skips the Open Targets fetch if `data/disease_genes.csv` already contains all three diseases. Delete the file to force a fresh fetch.

---

## Dependencies

```
pandas
networkx
numpy
matplotlib
requests
```

Install via `pip install -r requirements.txt`.

External APIs used (no authentication required for basic access):
- [Open Targets Platform API](https://platform.opentargets.org/api) — GraphQL
- [STRING DB API](https://string-db.org/cgi/help.pl) — REST
- [Semantic Scholar API](https://api.semanticscholar.org/) — REST (rate-limited; pipeline includes retry logic)
- [DisGeNET API](https://api.disgenet.com/) — REST (requires free API key; key included in `03_validate_fau.py`)

---

## Limitations and Planned Extensions

**Current limitations:**
- Generic PPI network (STRING) is context-agnostic — a synovial fibroblast and a T cell are treated as identical nodes
- RWR propagation weights are fixed by network topology, not learned from data
- Seed genes derived from Open Targets may reflect annotation bias toward well-studied proteins

**Planned extensions:**
1. Replace fixed-weight RWR with a Graph Convolutional Network (GCN) trained to learn evidence-specific edge weights
2. Construct cell-type-specific interactomes from single-cell RNA-seq data (scRNA-seq) to replace generic PPI edges
3. Incorporate GWAS polygenic signals as additional seed gene priors to reduce annotation bias
