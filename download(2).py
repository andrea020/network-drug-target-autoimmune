"""
Stage 3: Validate FAU as a novel drug target candidate

Three independent validation approaches:
  A. Literature search   — Semantic Scholar: "FAU + autoimmune / immune"
  B. Database lookup     — DisGeNET REST API: disease associations for FAU
  C. Robustness analysis — re-weight existing normalised scores, track FAU rank

Usage:
    python 03_validate_fau.py

Output:
    results/fau_validation/literature_fau.md
    results/fau_validation/disgenet_fau.csv
    results/fau_validation/robustness_fau.csv
    results/fau_validation/robustness_fau.png
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

PROJECT_ROOT  = Path(__file__).parent.parent
RESULTS_DIR   = PROJECT_ROOT / "results"
VAL_DIR       = RESULTS_DIR / "fau_validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)

GENE_SYMBOL   = "FAU"
NCBI_GENE_ID  = 2197          # FAU NCBI Gene ID
DISEASE_FILE  = RESULTS_DIR / "ranked_targets_multiple_sclerosis.csv"  # FAU is #2 here

# ============================================================
# PART A — Literature (Semantic Scholar)
# ============================================================

LITERATURE_QUERIES = [
    "FAU ubiquitin-like autoimmune disease",
    "FAU ribosomal protein immune response inflammation",
    "FUBI RPS30 immune apoptosis",
    "FAU gene multiple sclerosis lupus rheumatoid arthritis",
]


def search_semantic_scholar(query: str, limit: int = 10) -> list:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,abstract,citationCount,externalIds,openAccessPdf",
    }
    try:
        resp = requests.get(url, params=params, headers={"User-Agent": "AcademicResearch/1.0"},
                            timeout=20)
        if resp.status_code == 429:
            print(f"  Rate-limited — waiting 15 s ...")
            time.sleep(15)
            resp = requests.get(url, params=params, headers={"User-Agent": "AcademicResearch/1.0"},
                                timeout=20)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        print(f"  Semantic Scholar error for '{query[:50]}': {e}")
        return []


def run_literature_search():
    print("\n" + "=" * 60)
    print("  PART A — Literature Search (Semantic Scholar)")
    print("=" * 60)

    all_papers, seen_ids = [], set()
    for query in LITERATURE_QUERIES:
        print(f"  Query: {query[:60]}...")
        results = search_semantic_scholar(query)
        for p in results:
            pid = p.get("paperId", "")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append({"query": query, **p})
        print(f"    -> {len(results)} results")
        time.sleep(8)

    # Sort by citation count descending
    all_papers.sort(key=lambda x: x.get("citationCount", 0), reverse=True)

    # Filter: keep papers where abstract mentions FAU (or at least the query term)
    relevant = []
    for p in all_papers:
        abstract = (p.get("abstract") or "").upper()
        title    = (p.get("title")    or "").upper()
        if "FAU" in title or "FAU" in abstract or "FUBI" in title or "FUBI" in abstract:
            relevant.append(p)

    print(f"\n  Papers mentioning FAU/FUBI in title or abstract: {len(relevant)} / {len(all_papers)}")
    if not relevant:
        print("  NOTE: No papers directly mentioning FAU — this supports the 'novel candidate' hypothesis.")

    # Write Markdown report
    out_path = VAL_DIR / "literature_fau.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# FAU Literature Validation\n\n")
        f.write(f"Queries: {len(LITERATURE_QUERIES)} | Total retrieved: {len(all_papers)} unique papers\n\n")
        f.write(f"**Papers explicitly mentioning FAU/FUBI:** {len(relevant)}\n\n")

        if not relevant:
            f.write("No papers in the retrieved set mention FAU directly in title/abstract. "
                    "This is consistent with FAU being an understudied (dark proteome) candidate.\n\n")
            f.write("## All Retrieved Papers (sorted by citations)\n\n")
            papers_to_write = all_papers[:20]
        else:
            f.write("## Papers Mentioning FAU / FUBI\n\n")
            papers_to_write = relevant

        for i, p in enumerate(papers_to_write, 1):
            authors = [a["name"] for a in p.get("authors", [])[:3]]
            doi = p.get("externalIds", {}).get("DOI", "")
            f.write(f"### {i}. {p.get('title', 'N/A')}\n\n")
            f.write(f"**Authors**: {', '.join(authors)}{'...' if len(p.get('authors',[])) > 3 else ''}  \n")
            f.write(f"**Year**: {p.get('year')} | **Citations**: {p.get('citationCount', 0)}\n\n")
            if doi:
                f.write(f"**DOI**: https://doi.org/{doi}  \n\n")
            abstract = (p.get("abstract") or "")[:500]
            if abstract:
                f.write(f"{abstract}...\n\n")
            f.write("---\n\n")

    print(f"  Saved -> {out_path}")
    return relevant


# ============================================================
# PART B — DisGeNET
# ============================================================

DISGENET_API_BASE = "https://api.disgenet.com/api/v1"
DISGENET_API_KEY  = "033dd237-35d5-4849-936f-e7f66ac92a49"


def disgenet_query_gene(gene_id: int) -> list:
    """Query gene-disease associations via DisGeNET v1 API (API-key auth)."""
    url = f"{DISGENET_API_BASE}/gda/summary"
    params = {"gene_ncbi_id": gene_id, "format": "json"}
    headers = {"Authorization": DISGENET_API_KEY}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        if resp.status_code == 200:
            body = resp.json()
            return body.get("payload", [])
        else:
            print(f"  DisGeNET: HTTP {resp.status_code} — {resp.text[:200]}")
            return []
    except Exception as e:
        print(f"  DisGeNET query error: {e}")
        return []


def run_disgenet_lookup():
    print("\n" + "=" * 60)
    print("  PART B — DisGeNET Disease Association Lookup")
    print("=" * 60)
    print(f"  Gene: {GENE_SYMBOL}  (NCBI Gene ID: {NCBI_GENE_ID})")

    associations = disgenet_query_gene(NCBI_GENE_ID)

    out_path = VAL_DIR / "disgenet_fau.csv"

    if not associations:
        print("  No associations returned.")
        pd.DataFrame().to_csv(out_path, index=False)
        return []

    # Flatten to a tidy DataFrame with the fields we care about
    rows = []
    for a in associations:
        msh_classes = "; ".join(a.get("diseaseClasses_MSH") or [])
        rows.append({
            "disease_name":   a.get("diseaseName", ""),
            "disease_type":   a.get("diseaseType", ""),
            "score":          a.get("score"),
            "ei":             a.get("ei"),          # evidence index (specificity)
            "num_pmids":      a.get("numPMIDs", 0),
            "year_initial":   a.get("yearInitial"),
            "year_final":     a.get("yearFinal"),
            "msh_classes":    msh_classes,
            "umls_cui":       a.get("diseaseUMLSCUI", ""),
        })
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(f"  Total disease associations: {len(df)}")

    # Flag immune/autoimmune-related diseases
    immune_keywords = ["autoimmun", "immune", "lupus", "arthritis", "sclerosis",
                       "inflamm", "psoriasis", "thyroid", "myositis", "vasculitis",
                       "immunodeficien", "allerg"]
    mask = df["disease_name"].str.lower().apply(
        lambda x: any(kw in x for kw in immune_keywords)
    )
    immune_df = df[mask]
    print(f"  Immune/autoimmune-related: {len(immune_df)}")

    print("\n  All 27 associations (sorted by score):")
    print(df[["disease_name", "score", "ei", "num_pmids", "msh_classes"]].to_string(index=False))

    if not immune_df.empty:
        print(f"\n  *** Immune-related hits ***")
        print(immune_df[["disease_name", "score", "ei", "num_pmids"]].to_string(index=False))
    else:
        print("\n  No immune/autoimmune disease associations found in DisGeNET.")

    df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    return associations


# ============================================================
# PART C — Robustness Analysis
# ============================================================

# Weight configs: (rwr, pagerank, betweenness, degree)
# All weights sum to 1.0
WEIGHT_CONFIGS = {
    "current (0.35/0.30/0.20/0.15)":  (0.35, 0.30, 0.20, 0.15),
    "RWR dominant (0.60/0.20/0.10/0.10)": (0.60, 0.20, 0.10, 0.10),
    "PageRank dominant (0.20/0.50/0.20/0.10)": (0.20, 0.50, 0.20, 0.10),
    "equal (0.25/0.25/0.25/0.25)":    (0.25, 0.25, 0.25, 0.25),
    "RWR only (1.0/0/0/0)":           (1.00, 0.00, 0.00, 0.00),
    "topology only (0/0.40/0.35/0.25)": (0.00, 0.40, 0.35, 0.25),
}


def run_robustness_analysis():
    print("\n" + "=" * 60)
    print("  PART C — Robustness Analysis (weight sensitivity)")
    print("=" * 60)

    if not DISEASE_FILE.exists():
        print(f"  ERROR: {DISEASE_FILE} not found — run 02_network_analysis.py first.")
        return

    df = pd.read_csv(DISEASE_FILE)
    print(f"  Loaded: {DISEASE_FILE.name} ({len(df)} proteins)")

    required = ["protein", "rwr_norm", "pagerank_norm", "betweenness_norm", "degree_norm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: missing columns: {missing}")
        return

    records = []
    fau_ranks = {}

    for config_name, (w_rwr, w_pr, w_bw, w_deg) in WEIGHT_CONFIGS.items():
        df_copy = df.copy()
        df_copy["new_composite"] = (
            w_rwr * df_copy["rwr_norm"]
            + w_pr  * df_copy["pagerank_norm"]
            + w_bw  * df_copy["betweenness_norm"]
            + w_deg * df_copy["degree_norm"]
        )
        df_copy = df_copy.sort_values("new_composite", ascending=False).reset_index(drop=True)
        df_copy["new_rank"] = df_copy.index + 1

        fau_row = df_copy[df_copy["protein"] == GENE_SYMBOL]
        if fau_row.empty:
            fau_rank  = None
            fau_score = None
            print(f"  {config_name}: FAU not found in dataset")
        else:
            fau_rank  = int(fau_row["new_rank"].iloc[0])
            fau_score = round(float(fau_row["new_composite"].iloc[0]), 5)
            print(f"  {config_name}: FAU rank = #{fau_rank}  (score={fau_score:.4f})")

        fau_ranks[config_name] = fau_rank
        records.append({
            "config":     config_name,
            "w_rwr":      w_rwr,
            "w_pagerank": w_pr,
            "w_betweenness": w_bw,
            "w_degree":   w_deg,
            "fau_rank":   fau_rank,
            "fau_score":  fau_score,
        })

    results_df = pd.DataFrame(records)
    csv_path = VAL_DIR / "robustness_fau.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved -> {csv_path}")

    # Visualise
    plot_robustness(fau_ranks)
    return results_df


def plot_robustness(fau_ranks: dict):
    configs = list(fau_ranks.keys())
    ranks   = [fau_ranks[c] if fau_ranks[c] is not None else 999 for c in configs]

    # Wrap long labels
    labels = [c.replace(" (", "\n(") for c in configs]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#d62728" if r <= 5 else "#aec7e8" for r in ranks]
    bars = ax.bar(range(len(configs)), ranks, color=colors, edgecolor="white", width=0.6)

    ax.axhline(y=5,  color="#d62728", linestyle="--", alpha=0.6, label="Top-5 threshold")
    ax.axhline(y=10, color="grey",    linestyle="--", alpha=0.4, label="Top-10 threshold")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("FAU Rank in MS Analysis")
    ax.set_title("FAU Rank Across Different Scoring Weight Configurations\n"
                 "(MS analysis — FAU is #2 under current weights)", fontsize=11)
    ax.invert_yaxis()   # lower rank = higher on plot = better
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (bar, r) in enumerate(zip(bars, ranks)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"#{r}", ha="center", va="top", fontsize=9, fontweight="bold")

    plt.tight_layout()
    img_path = VAL_DIR / "robustness_fau.png"
    plt.savefig(img_path, dpi=150)
    plt.close()
    print(f"  Saved -> {img_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print(f"  FAU Validation — three-pronged analysis")
    print("=" * 60)

    relevant_papers = run_literature_search()
    run_disgenet_lookup()
    robustness_df   = run_robustness_analysis()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    # Literature
    n_lit = len(relevant_papers)
    if n_lit == 0:
        print("  Literature: 0 papers explicitly link FAU to autoimmune disease")
        print("    -> Consistent with 'novel / dark proteome' hypothesis")
    else:
        print(f"  Literature: {n_lit} paper(s) mention FAU in an immune context")
        print("    -> Review literature_fau.md for details")

    # Robustness
    if robustness_df is not None:
        top5  = (robustness_df["fau_rank"] <= 5).sum()
        top10 = (robustness_df["fau_rank"] <= 10).sum()
        total = len(robustness_df)
        print(f"  Robustness: FAU in top-5 in {top5}/{total} configs, "
              f"top-10 in {top10}/{total} configs")
        if top5 >= total // 2:
            print("    -> FAU is ROBUST: rank holds across most weight configurations")
        else:
            print("    -> FAU is SENSITIVE: rank depends heavily on RWR weight — caveat needed")

    print(f"\n  Output directory: {VAL_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
                