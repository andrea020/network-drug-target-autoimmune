"""
Stage 1: Fetch data from two public APIs (no registration required)

  - Open Targets GraphQL: disease-gene associations for autoimmune diseases
  - STRING DB REST API:   protein-protein interactions for those genes

Usage:
    pip install -r requirements.txt
    python 01_fetch_data.py

Output:
    data/disease_genes.csv    -- gene symbols + association scores per disease
    data/ppi_network.csv      -- protein A, protein B, interaction score (0-1)
"""

import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Open Targets EFO IDs for three well-studied autoimmune diseases
DISEASES = {
    "rheumatoid_arthritis": "EFO_0000685",
    "systemic_lupus":       "MONDO_0007915",
    "multiple_sclerosis":   "MONDO_0005301",
}

OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
STRING_PARTNERS_URL = "https://string-db.org/api/json/interaction_partners"
HUMAN_TAXON = 9606
STRING_SCORE_THRESHOLD = 700   # 0–1000; 700 = high-confidence
STRING_PARTNERS_LIMIT = 25     # top N partners per seed gene (brings in novel candidates)
OPENTARGETS_MIN_SCORE = 0.15   # 0–1 overall association score


# ---------------------------------------------------------------------------
# 1.  Open Targets → disease genes
# ---------------------------------------------------------------------------

_DISEASE_QUERY = """
query DiseaseTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    name
    associatedTargets(page: {index: 0, size: $size}) {
      rows {
        target {
          id
          approvedSymbol
          approvedName
        }
        score
      }
    }
  }
}
"""


def fetch_disease_genes(efo_id: str, disease_key: str, limit: int = 300) -> list[dict]:
    variables = {"efoId": efo_id, "size": limit}
    disease_data = None
    for attempt in range(3):
        try:
            resp = requests.post(
                OPENTARGETS_URL,
                json={"query": _DISEASE_QUERY, "variables": variables},
                timeout=40,
            )
            resp.raise_for_status()
            disease_data = resp.json()["data"]["disease"]
            if disease_data is not None:
                break
            time.sleep(5)
        except Exception as exc:
            print(f"  [Open Targets] Attempt {attempt+1} error for {disease_key}: {exc}")
            time.sleep(5)
    if disease_data is None:
        print(f"  [Open Targets] No data for {disease_key} after 3 attempts")
        return []
    rows = disease_data["associatedTargets"]["rows"]
    return [
        {
            "ensembl_id": r["target"]["id"],
            "symbol":     r["target"]["approvedSymbol"],
            "gene_name":  r["target"]["approvedName"],
            "ot_score":   round(r["score"], 4),
            "disease":    disease_key,
        }
        for r in rows
        if r["score"] >= OPENTARGETS_MIN_SCORE
    ]


def fetch_all_disease_genes() -> pd.DataFrame:
    print("--- Step 1: Open Targets disease-gene associations ---")
    records = []
    for key, efo_id in DISEASES.items():
        print(f"  {key} ({efo_id}) ...", end=" ", flush=True)
        genes = fetch_disease_genes(efo_id, key)
        print(f"{len(genes)} genes")
        records.extend(genes)
        time.sleep(1)

    df = pd.DataFrame(records).drop_duplicates(subset=["symbol", "disease"])
    out = DATA_DIR / "disease_genes.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} rows → {out}\n")
    return df


# ---------------------------------------------------------------------------
# 2.  STRING DB → PPI network
# ---------------------------------------------------------------------------

def _string_network_chunk(symbols: list[str]) -> list[dict]:
    params = {
        "identifiers":    "%0d".join(symbols),  # STRING uses %0d as separator
        "species":        HUMAN_TAXON,
        "required_score": STRING_SCORE_THRESHOLD,
        "limit":          STRING_PARTNERS_LIMIT,
        "caller_identity": "phd_network_biology_project",
    }
    try:
        resp = requests.post(STRING_PARTNERS_URL, data=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"  [STRING] Chunk error: {exc}")
        return []


def fetch_ppi_network(gene_symbols: list[str], chunk_size: int = 100) -> pd.DataFrame:
    print("--- Step 2: STRING DB protein-protein interactions ---")
    all_interactions = []

    for i in range(0, len(gene_symbols), chunk_size):
        chunk = gene_symbols[i : i + chunk_size]
        interactions = _string_network_chunk(chunk)
        all_interactions.extend(interactions)
        print(f"  Chunk {i // chunk_size + 1}/{-(-len(gene_symbols) // chunk_size)}"
              f" ({len(chunk)} genes) → {len(interactions)} interactions")
        time.sleep(1)

    if not all_interactions:
        print("  No interactions returned. Check gene symbols or STRING API.")
        return pd.DataFrame(columns=["protein_a", "protein_b", "score"])

    df = (
        pd.DataFrame(all_interactions)
        [["preferredName_A", "preferredName_B", "score"]]
        .rename(columns={"preferredName_A": "protein_a", "preferredName_B": "protein_b"})
        .assign(score=lambda d: d["score"].astype(float))  # STRING returns score as 0–1 float
        .drop_duplicates()
        .reset_index(drop=True)
    )

    out = DATA_DIR / "ppi_network.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} interactions → {out}\n")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    disease_path = DATA_DIR / "disease_genes.csv"
    if disease_path.exists():
        existing = pd.read_csv(disease_path)
        if set(existing["disease"].unique()) >= set(DISEASES.keys()):
            print(f"Skipping Open Targets fetch — all 3 diseases already present.")
            disease_df = existing
        else:
            disease_df = fetch_all_disease_genes()
    else:
        disease_df = fetch_all_disease_genes()

    gene_symbols = disease_df["symbol"].dropna().unique().tolist()
    print(f"Unique genes across all diseases: {len(gene_symbols)}")

    ppi_df = fetch_ppi_network(gene_symbols)

    print("=== Summary ===")
    print(f"  Disease-gene associations : {len(disease_df)} rows")
    print(f"  PPI interactions          : {len(ppi_df)} rows")
    print(f"  Unique proteins in network: {len(set(ppi_df['protein_a']) | set(ppi_df['protein_b']))}")
    print("\nNext step: run 02_network_analysis.py")


if __name__ == "__main__":
    main()
        