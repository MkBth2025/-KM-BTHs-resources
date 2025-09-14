import os
import re
import math
import itertools
import logging
import warnings
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio as levenshtein_ratio

# Optional BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except Exception:
    BM25_AVAILABLE = False
    BM25Okapi = None

# Optional SBERT
EMBEDDINGS_ENABLED = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    EMBEDDINGS_ENABLED = False
    SentenceTransformer = None

from scipy.stats import mannwhitneyu, shapiro, norm, ttest_ind
from statsmodels.stats.power import TTestIndPower
from math import sqrt

# ---------- Logging setup ----------
for h in list(logging.root.handlers):
    logging.root.removeHandler(h)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("similarity_pipeline")

warnings.filterwarnings("ignore")

CONFIG = {
    "INPUT_FILE": "data/test_dataset1.xlsx",
    "OUTPUT_PAIRWISE_FILE": "report/02_Similarity_Analysis.csv",
    "OUTPUT_STATS_FILE": "report/02_Similarity_Statistical_Analysis.csv",
    "WEIGHTS": {
        "JACCARD": 0.25,
        "LEVENSHTEIN": 0.25,
        "TFIDF": 0.30,
        "BM25": 0.10,
        "SBERT": 0.10
    },
    "NGRAM_RANGE": (1, 2),
    "MAX_FEATURES": 8000,
    "ALPHA": 0.05,
    "RANGE_START": None,
    "RANGE_END": None,
    "MIN_COLUMNS_PER_GROUP": 2,  # Minimum columns required to form a group
}

def auto_detect_column_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Automatically detect column groups based on first 3 characters"""
    prefix_to_columns = {}
    
    # Group all columns by their first 3 characters
    for col in df.columns:
        if len(col) >= 3:  # Only process columns with at least 3 characters
            prefix = col[:3].upper()  # Use uppercase for consistency
            if prefix not in prefix_to_columns:
                prefix_to_columns[prefix] = []
            prefix_to_columns[prefix].append(col)
    
    # Filter groups to only include those with minimum required columns
    valid_groups = {
        prefix: cols for prefix, cols in prefix_to_columns.items() 
        if len(cols) >= CONFIG["MIN_COLUMNS_PER_GROUP"]
    }
    
    # Log detected groups
    logger.info(f"Detected column groups:")
    for prefix, cols in valid_groups.items():
        logger.info(f"  {prefix}: {cols}")
    
    if not valid_groups:
        logger.warning("No valid column groups detected (each group needs at least 2 columns)")
    
    return valid_groups

def extract_activities(text: Any) -> List[str]:
    if pd.isna(text) or not isinstance(text, str):
        return []
    txt = text.lower()
    spans = re.findall(r":\s*([^;]+)\s*;", txt)
    tokens = []
    for s in spans:
        s = re.sub(r"[^\w\s]", " ", s)
        tokens.extend(re.findall(r"\b\w{2,}\b", s))
    if not tokens:
        clean = re.sub(r"[^\w\s]", " ", txt)
        tokens = re.findall(r"\b\w{2,}\b", clean)
    return tokens

def jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def tfidf_cosine(doc1: str, doc2: str, ngram_range=(1,2), max_features=8000) -> float:
    vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, lowercase=True)
    X = vec.fit_transform([doc1 or "", doc2 or ""])
    return float(cosine_similarity(X[0], X[1])[0, 0])

def levenshtein_similarity(a: Any, b: Any) -> float:
    if pd.isna(a) and pd.isna(b):
        return 1.0
    if pd.isna(a) or pd.isna(b):
        return 0.0
    sa, sb = str(a).strip(), str(b).strip()
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(levenshtein_ratio(sa, sb))

def bm25_similarity(doc1: str, doc2: str) -> float:
    if not BM25_AVAILABLE:
        return np.nan
    t1 = extract_activities(doc1)
    t2 = extract_activities(doc2)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    corpus = [t1, t2]
    bm25 = BM25Okapi(corpus)
    s12 = bm25.get_scores(t1)[1]
    s21 = bm25.get_scores(t2)[0]
    max_s = max(s12, s21, 1e-9)
    sim = float((s12 + s21) / (2.0 * max_s))
    return max(0.0, min(1.0, sim))

_SBERT_MODEL = None
def load_sbert():
    global _SBERT_MODEL
    if not EMBEDDINGS_ENABLED or SentenceTransformer is None:
        return None
    if _SBERT_MODEL is None:
        logger.info("Loading SBERT model...")
        _SBERT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("SBERT model loaded.")
    return _SBERT_MODEL

def sbert_cosine(doc1: str, doc2: str) -> float:
    if not EMBEDDINGS_ENABLED or SentenceTransformer is None:
        return np.nan
    model = load_sbert()
    emb = model.encode([doc1 or "", doc2 or ""], normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    sim = float(np.dot(emb[0], emb[1]))
    return (sim + 1.0) / 2.0

def weighted_similarity(jacc: float, lev: float, tfidf_s: float, bm25_s: float, sbert_s: float, weights: Dict[str,float]) -> float:
    comps, wts = [], []
    if not np.isnan(jacc): comps.append(jacc); wts.append(weights["JACCARD"])
    if not np.isnan(lev): comps.append(lev); wts.append(weights["LEVENSHTEIN"])
    if not np.isnan(tfidf_s): comps.append(tfidf_s); wts.append(weights["TFIDF"])
    if not np.isnan(bm25_s) and BM25_AVAILABLE: comps.append(bm25_s); wts.append(weights["BM25"])
    if not np.isnan(sbert_s): comps.append(sbert_s); wts.append(weights["SBERT"])
    if not comps:
        return np.nan
    return float(np.average(comps, weights=wts))

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    s_pooled = sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    if s_pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / s_pooled

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    i = j = gt = lt = 0
    while i < nx and j < ny:
        if x_sorted[i] > y_sorted[j]:
            gt += (nx - i)
            j += 1
        elif x_sorted[i] < y_sorted[j]:
            lt += (ny - j)
            i += 1
        else:
            i += 1
            j += 1
    return (gt - lt) / (nx * ny)

def mean_confidence_interval(data: np.ndarray, alpha=0.05) -> Tuple[float, float]:
    data = np.asarray(data)
    n = len(data)
    if n < 2:
        return (np.nan, np.nan)
    m = np.mean(data)
    s = np.std(data, ddof=1)
    z = norm.ppf(1 - alpha/2.0)
    half = z * s / math.sqrt(n)
    return (m - half, m + half)

def compute_power_ttest_ind(x: np.ndarray, y: np.ndarray, alpha=0.05) -> float:
    d = abs(cohen_d(x, y))
    if np.isnan(d) or d <= 0:
        return np.nan
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    analysis = TTestIndPower()
    try:
        return float(analysis.power(effect_size=d, nobs1=nx, ratio=ny/max(ny, 1), alpha=alpha, alternative="two-sided"))
    except Exception:
        return np.nan

def validate_input_file(path: str) -> bool:
    try:
        pd.read_excel(path, nrows=1)
        return True
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return False

def read_input(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def pairwise_similarity_row_all_columns(row: pd.Series, all_columns: List[str], valid_groups: Dict[str, List[str]]) -> Dict[str, Any]:
    """Generate pairwise similarities for all possible column combinations, but only calculate for valid groups"""
    res = {"Index": int(row.name)}
    texts = {c: ("" if pd.isna(row[c]) else str(row[c])) for c in all_columns}
    tokens = {c: extract_activities(texts[c]) for c in all_columns}
    
    # Create a mapping of column to its group prefix
    col_to_prefix = {}
    for prefix, cols in valid_groups.items():
        for col in cols:
            col_to_prefix[col] = prefix
    
    # Process all possible column pairs
    for (c1, c2) in itertools.combinations(all_columns, 2):
        key = f"{c1}_{c2}"
        
        # Only calculate similarities if both columns belong to the same valid group
        c1_prefix = col_to_prefix.get(c1)
        c2_prefix = col_to_prefix.get(c2)
        
        if (c1_prefix is not None and c2_prefix is not None and c1_prefix == c2_prefix):
            t1, t2 = tokens[c1], tokens[c2]
            s1, s2 = texts[c1], texts[c2]
            jacc = jaccard_similarity(t1, t2)
            lev = levenshtein_similarity(s1, s2)
            tfidf_s = tfidf_cosine(s1, s2, ngram_range=CONFIG["NGRAM_RANGE"], max_features=CONFIG["MAX_FEATURES"])
            bm25_s = bm25_similarity(s1, s2)
            sbert_s = sbert_cosine(s1, s2) if EMBEDDINGS_ENABLED and SentenceTransformer is not None else np.nan
            w = weighted_similarity(jacc, lev, tfidf_s, bm25_s, sbert_s, CONFIG["WEIGHTS"])
            
            res[f"{key}_Jaccard"] = round(jacc, 4)
            res[f"{key}_Levenshtein"] = round(lev, 4)
            res[f"{key}_TFIDF"] = round(tfidf_s, 4)
            if BM25_AVAILABLE and not np.isnan(bm25_s):
                res[f"{key}_BM25"] = round(bm25_s, 4)
            if EMBEDDINGS_ENABLED and SentenceTransformer is not None and not np.isnan(sbert_s):
                res[f"{key}_SBERT"] = round(sbert_s, 4)
            res[f"{key}_MEAN"] = round(np.nanmean([
                jacc, lev, tfidf_s, (bm25_s if BM25_AVAILABLE else np.nan), sbert_s
            ]), 4)
            res[f"{key}_Weighted"] = round(w, 4)
        else:
            # Set empty values for pairs not in the same valid group
            res[f"{key}_Jaccard"] = np.nan
            res[f"{key}_Levenshtein"] = np.nan
            res[f"{key}_TFIDF"] = np.nan
            if BM25_AVAILABLE:
                res[f"{key}_BM25"] = np.nan
            if EMBEDDINGS_ENABLED and SentenceTransformer is not None:
                res[f"{key}_SBERT"] = np.nan
            res[f"{key}_MEAN"] = np.nan
            res[f"{key}_Weighted"] = np.nan
    
    return res

def build_pairwise_output_all_columns(df: pd.DataFrame, all_columns: List[str], valid_groups: Dict[str, List[str]]) -> pd.DataFrame:
    """Build pairwise output for all columns but only calculate similarities for valid groups"""
    records = []
    for local_idx, (idx, row) in enumerate(df.iterrows()):
        if local_idx % 50 == 0:
            logger.info(f"Processing row {local_idx+1}/{len(df)} (original index: {idx})")
        row_res = pairwise_similarity_row_all_columns(row, all_columns, valid_groups)
        records.append(row_res)
    return pd.DataFrame(records)

def split_intra_inter_pairs(columns: List[str], valid_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    # Create a mapping of column to its group prefix
    col_to_prefix = {}
    for prefix, cols in valid_groups.items():
        for col in cols:
            col_to_prefix[col] = prefix
    
    pair_keys, intra_keys, inter_keys = [], [], []
    for (c1, c2) in itertools.combinations(columns, 2):
        key = f"{c1}_{c2}"
        pair_keys.append(key)
        p1 = col_to_prefix.get(c1)
        p2 = col_to_prefix.get(c2)
        if p1 is not None and p2 is not None and p1 == p2:
            intra_keys.append(key)
        else:
            inter_keys.append(key)
    return {"all": pair_keys, "intra": intra_keys, "inter": inter_keys}

def compute_column_stats(pairwise_df: pd.DataFrame, columns: List[str], metric_name: str, valid_groups: Dict[str, List[str]], alpha=0.05) -> pd.DataFrame:
    keys = split_intra_inter_pairs(columns, valid_groups)
    all_pairs = keys["all"]
    intra_pairs = set(keys["intra"])
    inter_pairs = set(keys["inter"])

    rows = []
    for col in columns:
        matched_pairs = [k for k in all_pairs if k.startswith(f"{col}_") or k.endswith(f"_{col}")]
        vals_all, vals_intra, vals_inter = [], [], []
        for k in matched_pairs:
            colname = f"{k}_{metric_name}"
            if colname in pairwise_df.columns:
                v = pairwise_df[colname].values
                v = v[~pd.isna(v)]
                if v.size:
                    vals_all.extend(v.tolist())
                    if k in intra_pairs:
                        vals_intra.extend(v.tolist())
                    if k in inter_pairs:
                        vals_inter.extend(v.tolist())

        vals_all = np.array(vals_all, dtype=float)
        vals_intra = np.array(vals_intra, dtype=float)
        vals_inter = np.array(vals_inter, dtype=float)

        mean_val = float(np.mean(vals_all)) if vals_all.size else np.nan
        std_val = float(np.std(vals_all, ddof=1)) if vals_all.size > 1 else np.nan

        shapiro_p = np.nan
        if vals_all.size >= 3:
            try:
                shapiro_p = float(shapiro(vals_all)[1])
            except Exception:
                shapiro_p = np.nan

        ci_low, ci_high = mean_confidence_interval(vals_all, alpha=alpha) if vals_all.size >= 2 else (np.nan, np.nan)

        z_stat = np.nan
        p_value = np.nan
        effect_size = np.nan
        power_1m_beta = np.nan
        mw_p = np.nan

        if vals_intra.size >= 2 and vals_inter.size >= 2:
            normal = False
            try:
                if vals_intra.size >= 3 and vals_inter.size >= 3:
                    p1 = shapiro(vals_intra)[1]
                    p2 = shapiro(vals_inter)[1]
                    normal = (p1 > alpha) and (p2 > alpha)
            except Exception:
                normal = False

            if normal:
                t_stat, p_val = ttest_ind(vals_intra, vals_inter, equal_var=False)
                z_stat = float(t_stat)
                p_value = float(p_val)
                effect_size = float(cohen_d(vals_intra, vals_inter))
                power_1m_beta = compute_power_ttest_ind(vals_intra, vals_inter, alpha=alpha)
                try:
                    mw_stat, mw_pv = mannwhitneyu(vals_intra, vals_inter, alternative="two-sided")
                    mw_p = float(mw_pv)
                except Exception:
                    mw_p = np.nan
            else:
                try:
                    mw_stat, mw_pv = mannwhitneyu(vals_intra, vals_inter, alternative="two-sided")
                    mw_p = float(mw_pv)
                except Exception:
                    mw_p = np.nan
                try:
                    nx, ny = len(vals_intra), len(vals_inter)
                    U = mw_stat
                    mu_U = nx * ny / 2.0
                    sigma_U = math.sqrt(nx * ny * (nx + ny + 1) / 12.0)
                    z_stat = float((U - mu_U) / sigma_U) if sigma_U > 0 else np.nan
                except Exception:
                    z_stat = np.nan
                p_value = mw_p
                effect_size = float(cliffs_delta(vals_intra, vals_inter))
                try:
                    d_approx = abs(effect_size) * 1.1
                    if d_approx > 0:
                        analysis = TTestIndPower()
                        power_1m_beta = float(
                            analysis.power(
                                effect_size=d_approx,
                                nobs1=len(vals_intra),
                                ratio=len(vals_inter)/max(len(vals_inter), 1),
                                alpha=alpha,
                                alternative="two-sided",
                            )
                        )
                except Exception:
                    power_1m_beta = np.nan

        rows.append({
            "Column": col,
            "Metric": metric_name,
            "Mean": round(mean_val, 6) if not np.isnan(mean_val) else np.nan,
            "Std Dev": round(std_val, 6) if not np.isnan(std_val) else np.nan,
            "Shapiro-Wilk p": round(shapiro_p, 6) if not np.isnan(shapiro_p) else np.nan,
            "Z-stat": round(z_stat, 6) if not np.isnan(z_stat) else np.nan,
            "P-value": round(p_value, 6) if not np.isnan(p_value) else np.nan,
            "Effect Size": round(effect_size, 6) if not np.isnan(effect_size) else np.nan,
            "Power (1-β)": round(power_1m_beta, 6) if not np.isnan(power_1m_beta) else np.nan,
            "Mann-Whitney U p": round(mw_p, 6) if not np.isnan(mw_p) else np.nan,
            "95% CI Lower": round(ci_low, 6) if not np.isnan(ci_low) else np.nan,
            "95% CI Upper": round(ci_high, 6) if not np.isnan(ci_high) else np.nan,
            "N_all": len(vals_all),
            "N_intra": len(vals_intra),
            "N_inter": len(vals_inter),
        })

    return pd.DataFrame(rows)

def apply_range(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is None and end is None:
        return df
    n = len(df)
    s = 0 if start is None else max(0, int(start))
    e = n if end is None else min(n, int(end))
    if s >= e:
        logger.warning(f"Invalid range: start={s}, end={e}. Processing full dataset.")
        return df
    logger.info(f"Applying range iloc[{s}:{e}] over {n} total rows.")
    return df.iloc[s:e].copy()

def suffix_with_range(base_name: str, start, end) -> str:
    root, ext = os.path.splitext(base_name)
    if start is None and end is None:
        return base_name
    s = "" if start is None else str(start)
    e = "" if end is None else str(end)
    return f"{root}_range_{s}_{e}{ext}"

def main():
    print("Starting similarity pipeline...")
    if not validate_input_file(CONFIG["INPUT_FILE"]):
        return

    df_full = read_input(CONFIG["INPUT_FILE"])
    
    # Automatically detect column groups based on first 3 characters
    valid_groups = auto_detect_column_groups(df_full)
    
    if not valid_groups:
        logger.error("No valid column groups detected. Please check your data.")
        return
    
    # Get all columns from valid groups
    all_columns = []
    for cols in valid_groups.values():
        all_columns.extend(cols)
    
    df = apply_range(df_full, CONFIG["RANGE_START"], CONFIG["RANGE_END"])
    logger.info(f"Number of rows to process: {len(df)}")

    if EMBEDDINGS_ENABLED and SentenceTransformer is not None:
        try:
            load_sbert()
        except Exception as e:
            logger.warning(f"SBERT not available: {e}")

    # Generate single pairwise output with all columns
    logger.info("Computing pairwise similarities for all detected groups...")
    pairwise_df = build_pairwise_output_all_columns(df, all_columns, valid_groups)
    
    # Save pairwise results
    output_file = suffix_with_range(CONFIG["OUTPUT_PAIRWISE_FILE"], CONFIG["RANGE_START"], CONFIG["RANGE_END"])
    pairwise_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info(f"Pairwise results saved to '{output_file}'.")

    # Compute statistics for each detected group
    stats_all = []
    
    for prefix, cols in valid_groups.items():
        logger.info(f"Computing statistics for {prefix} group...")
        
        metrics_for_stats = ["Jaccard", "Levenshtein", "TFIDF", "MEAN", "Weighted"]
        if BM25_AVAILABLE:
            metrics_for_stats.append("BM25")
        if EMBEDDINGS_ENABLED and SentenceTransformer is not None and any(c.endswith("_SBERT") for c in pairwise_df.columns):
            metrics_for_stats.append("SBERT")

        for m in metrics_for_stats:
            logger.info(f"Computing stats for {prefix} group with metric {m}")
            stat_df = compute_column_stats(pairwise_df, cols, m, valid_groups, alpha=CONFIG["ALPHA"])
            stat_df["Prefix_Group"] = prefix
            stats_all.append(stat_df)

    # Save statistics
    if stats_all:
        df_all_stats = pd.concat(stats_all, ignore_index=True)
        stats_file = suffix_with_range(CONFIG["OUTPUT_STATS_FILE"], CONFIG["RANGE_START"], CONFIG["RANGE_END"])
        df_all_stats.to_csv(stats_file, index=False, encoding="utf-8-sig")
        logger.info(f"Statistics saved to '{stats_file}'.")

if __name__ == "__main__":
    main()
