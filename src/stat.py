# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, skew, kurtosis, probplot
from matplotlib.backends.backend_pdf import PdfPages

# ---------- Helpers ----------
def _safe_to_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns and not pd.api.types.is_numeric_dtype(d[c]):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def extract_group(col: str, source_file: str) -> str:
    """Extract main group identifier depending on file type."""
    if ("03_Symantic_promt_code_Analysis_SP" in source_file or
        "03_Symantic_promt_code_Analysis_MP" in source_file):
        if "prompt_" in col:
            try:
                return col.split("prompt_")[1][:3]
            except Exception:
                return col[:3]
        else:
            return col[:3]
    else:
        return col[:3]

# ---------- Core analysis ----------
def compute_stats(df: pd.DataFrame, cols, source_file: str) -> pd.DataFrame:
    d = _safe_to_numeric(df[cols], cols)
    results = []
    for col in cols:
        group = extract_group(col, source_file)
        vals = d[col].dropna().values

        # --- NEW: Scale percentages 0-100 → 0-1
        if len(vals) > 0 and np.nanmin(vals) >= 0 and np.nanmax(vals) <= 100:
            vals = vals / 100.0

        if len(vals) == 0:
            results.append({
                "Feature": col, "Group": group,
                "Mean": np.nan, "Std Dev": np.nan, "Std/Mean": np.nan,
                "Shapiro-Wilk p": np.nan, "Skewness": np.nan,
                "Kurtosis": np.nan, "Missing %": 1.0
            })
            continue

        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1))
        miss_rate = float(d[col].isna().mean())
        try:
            shapiro_p = float(shapiro(vals).pvalue) if len(vals) > 3 else np.nan
        except Exception:
            shapiro_p = np.nan
        skew_val = float(skew(vals)) if len(vals) > 3 else np.nan
        kurt_val = float(kurtosis(vals)) if len(vals) > 3 else np.nan
        ratio = np.inf if np.isnan(mean_val) or mean_val == 0 else abs(std_val / mean_val)

        results.append({
            "Feature": col, "Group": group,
            "Mean": mean_val, "Std Dev": std_val, "Std/Mean": ratio,
            "Shapiro-Wilk p": shapiro_p, "Skewness": skew_val,
            "Kurtosis": kurt_val, "Missing %": miss_rate
        })
    return pd.DataFrame(results)

def compute_group_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = ["Mean","Std Dev","Std/Mean","Shapiro-Wilk p","Skewness","Kurtosis","Missing %"]
    all_groups = stats_df["Group"].unique()
    grouped = stats_df.groupby("Group")[agg_cols].mean().reindex(all_groups).reset_index()
    return grouped

# ---------- Ranking ----------
def add_ranking(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def score_row(row):
        score = 0
        if pd.notna(row["Std/Mean"]) and row["Std/Mean"] < 0.5: score += 1
        if pd.notna(row["Shapiro-Wilk p"]) and row["Shapiro-Wilk p"] >= 0.05: score += 1
        if pd.notna(row["Missing %"]):
            if row["Missing %"] < 0.05: score += 2
        if pd.notna(row.get("Skewness")) and pd.notna(row.get("Kurtosis")):
            if abs(row["Skewness"]) < 1 and abs(row["Kurtosis"]) < 3: score += 1
        return score

    df["Score"] = df.apply(score_row, axis=1)
    df.sort_values(by=["Score","Shapiro-Wilk p"], ascending=[False, False], inplace=True)
    df["Ranking"] = range(1, len(df)+1)
    return df

# ---------- Plotting ----------
def plot_summary_page(ranking_df, group_df, pdf: PdfPages, dataset_name: str):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.axis("off")
    text_lines = [f"📊 Statistical Summary Report for {dataset_name}", ""]
    text_lines.append("🏆 Top 10 Ranked Features:")
    for _, row in ranking_df.head(10).iterrows():
        text_lines.append(
            f"{int(row['Ranking'])}. {row['Feature']} "
            f"(Mean={row['Mean']:.2f}, Std={row['Std Dev']:.2f}, "
            f"Shapiro p={row['Shapiro-Wilk p']:.3f}, Score={row['Score']})"
        )
    text_lines.append("")
    text_lines.append("📊 All Main Groups Summary:")
    for _, row in group_df.iterrows():
        text_lines.append(
            f"Group {row['Group']} → Mean={row['Mean']:.2f}, Std={row['Std Dev']:.2f}, "
            f"Skew={row['Skewness']:.2f}, Kurt={row['Kurtosis']:.2f}, "
            f"Missing%={row['Missing %']*100:.2f}, Rank={row['Ranking']}"
        )
    ax.text(0,1,"\n".join(text_lines), va="top", ha="left", fontsize=11, wrap=True)
    pdf.savefig(fig); plt.close(fig)

def plot_normality(df: pd.DataFrame, col: str, pdf: PdfPages):
    vals = df[col].dropna().values
    if len(vals)<3: return
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    sns.histplot(vals, kde=True, ax=axes[0], color="skyblue", edgecolor="black")
    mu, sigma = np.mean(vals), np.std(vals)
    x = np.linspace(min(vals), max(vals), 200)
    axes[0].plot(x,(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2), color="red", lw=2)
    axes[0].set_title(f"{col} Histogram")
    probplot(vals, dist="norm", plot=axes[1])
    axes[1].set_title(f"{col} Q-Q Plot")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

def plot_group_summary(group_df: pd.DataFrame, group_name: str, pdf: PdfPages):
    if group_df.empty: return
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    metrics = ["Mean","Std Dev","Skewness","Kurtosis"]
    vals = group_df.loc[group_df["Group"]==group_name, metrics].values.flatten()
    sns.barplot(x=metrics,y=vals,ax=axes[0],palette="coolwarm"); axes[0].set_title(f"{group_name} - Summary Stats")
    metrics2 = ["Shapiro-Wilk p","Missing %"]
    vals2 = group_df.loc[group_df["Group"]==group_name, metrics2].values.flatten()
    sns.barplot(x=metrics2,y=vals2,ax=axes[1],palette="viridis"); axes[1].set_title(f"{group_name} - Quality Metrics")
    ratio = group_df.loc[group_df["Group"]==group_name,"Std/Mean"].values[0]
    axes[2].bar(["Std/Mean"],[ratio],color="orange"); axes[2].set_title(f"{group_name} - Variability")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

# ---------- Merge + Summary ----------
def interpret_group(row):
    interp=[]
    interp.append("Stable variance" if pd.notna(row["Std/Mean"]) and row["Std/Mean"]<0.5 else "High variance")
    interp.append("Likely normal distribution" if pd.notna(row["Shapiro-Wilk p"]) and row["Shapiro-Wilk p"]>=0.05 else "Non-normal distribution")
    if pd.notna(row["Missing %"]):
        if row["Missing %"]<0.05: interp.append("Low missingness")
        elif row["Missing %"]<0.2: interp.append("Moderate missingness")
        else: interp.append("High missingness")
    if pd.notna(row["Skewness"]) and pd.notna(row["Kurtosis"]):
        if abs(row["Skewness"])<1 and abs(row["Kurtosis"])<3:
            interp.append("Moderate shape")
        else:
            interp.append("Irregular shape")
    return "; ".join(interp)

def merge_and_summarize(outdir: Path):
    files=[
        outdir/"01_Syntatic_Analysis"/"01_Syntatic_Analysis_stat_all.csv",
        outdir/"02_Similarity_Analysis"/"02_Similarity_Analysis_stat_all.csv",
        outdir/"03_Symantic_promt_code_Analysis_SP"/"03_Symantic_promt_code_Analysis_SP_stat_all.csv",
        outdir/"03_Symantic_promt_code_Analysis_MP"/"03_Symantic_promt_code_Analysis_MP_stat_all.csv",
    ]
    dfs=[]
    for f in files:
        if f.exists():
            df=pd.read_csv(f); df["Source"]=f.stem; dfs.append(df)
        else:
            print(f"[WARN] Missing file: {f}")
    if not dfs:
        print("[ERROR] No stat_all files found for merge."); return

    merged=pd.concat(dfs,ignore_index=True)
    merged_ranked=add_ranking(merged)
    merged_out=outdir/"Analysis_stat_all.csv"; merged_ranked.to_csv(merged_out,index=False)
    print(f"[OK] Wrote merged CSV with ranking: {merged_out}")

    group_cols=["Mean","Std Dev","Std/Mean","Shapiro-Wilk p","Skewness","Kurtosis","Missing %"]
    summary=merged.groupby("Group")[group_cols].mean().reset_index()
    summary_ranked=add_ranking(summary)
    summary_out=outdir/"Summery_Analysis_stat_all.csv"; summary_ranked.to_csv(summary_out,index=False)
    print(f"[OK] Wrote summary CSV with ranking: {summary_out}")

    lines=["📊 Overall Statistical Summary (merged from 4 analyses)\n"]
    for _,row in summary_ranked.iterrows():
        lines.append(
            f"Rank {row['Ranking']} (Score={row['Score']}) → Group {row['Group']}: "
            f"Mean={row['Mean']:.2f}, Std={row['Std Dev']:.2f}, "
            f"Std/Mean={row['Std/Mean']:.2f}, Shapiro-Wilk p={row['Shapiro-Wilk p']:.3f}, "
            f"Skew={row['Skewness']:.2f}, Kurt={row['Kurtosis']:.2f}, "
            f"Missing%={row['Missing %']*100:.2f} → {interpret_group(row)}"
        )
    txt_out=outdir/"Summery_Analysis_stat_all.txt"; txt_out.write_text("\n".join(lines),encoding="utf-8")
    print(f"[OK] Wrote interpretation TXT with ranking: {txt_out}")

# ---------- CLI ----------
def main():
    ap=argparse.ArgumentParser(description="Statistical Analysis using stat.yaml rules")
    ap.add_argument("--stat-yaml",required=True,help="Path to stat.yaml file")
    ap.add_argument("--report-dir",default="report",help="Input report directory containing CSVs")
    ap.add_argument("--outdir",default="report/Stat",help="Output directory for stats")
    args=ap.parse_args()

    report_dir=Path(args.report_dir)
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    with open(args.stat_yaml,"r",encoding="utf-8") as f: config=yaml.safe_load(f)

    for entry in config.get("statistics",[]):
        fname=entry["file"]; suffixes=entry.get("columns",{}).get("include",[])
        inpath=report_dir/fname; outsubdir=outdir/Path(fname).stem; outsubdir.mkdir(parents=True,exist_ok=True)
        print(f"[INFO] Processing {inpath}")
        if not inpath.exists():
            print(f"[WARN] File not found: {inpath}, skipping."); continue
        try: df=pd.read_csv(inpath); print(f"[DEBUG] {fname} loaded with shape {df.shape}")
        except Exception as e: print(f"[ERROR] Could not read {inpath}: {e}"); continue
        cols=[]
        for suff in suffixes: cols.extend([c for c in df.columns if c.endswith(suff)])
        if not cols: print(f"[WARN] No matching columns found in {fname}"); continue
        print(f"[INFO] Columns used for {fname}: {cols}")
        try:
            stats_df=compute_stats(df,cols,fname)
            stats_ranked=add_ranking(stats_df)
            stats_ranked.to_csv(outsubdir/(Path(fname).stem+"_Stat.csv"),index=False)
            group_df=compute_group_stats(stats_df)
            group_ranked=add_ranking(group_df)
            group_ranked.to_csv(outsubdir/(Path(fname).stem+"_stat_all.csv"),index=False)
            pdf_path=outsubdir/(Path(fname).stem+"_plots.pdf")
            with PdfPages(pdf_path) as pdf:
                plot_summary_page(stats_ranked,group_ranked,pdf,Path(fname).stem)
                for col in stats_df["Feature"]: plot_normality(df,col,pdf)
                for g in group_ranked["Group"]: plot_group_summary(group_ranked,g,pdf)
        except Exception as e: print(f"[ERROR] Could not compute stats for {fname}: {e}")

    merge_and_summarize(outdir)

if __name__=="__main__":
    main()
