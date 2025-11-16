#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EDA visualizations for Kinect skeleton gesture dataset (local-friendly).
- No seaborn, matplotlib only.
- Saves figures (PNGs) and summary tables (CSVs).
- One chart per plot; can be many if you enable per-class/per-performer histograms.

Usage examples:
  python EDA_visual.py --input "/path/to/gesture_database.csv" --outdir eda_out
  python EDA_visual.py --input gesture_database.csv --outdir eda_out --per-class --per-performer

Notes:
  - Assumes 60 coordinate columns + metadata: sequence_id, performer, gesture_type, session.
  - Handles mixed string/int session values.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe; comment out if you want interactive windows
import matplotlib.pyplot as plt


def detect_coord_columns(df: pd.DataFrame) -> list[str]:
    meta = {"sequence_id", "performer", "gesture_type", "session"}
    cols = [c for c in df.columns if c not in meta]
    return cols


def coerce_session_dtype(df: pd.DataFrame, prefer_numeric: bool = True) -> None:
    if "session" not in df.columns:
        return
    if prefer_numeric:
        s = pd.to_numeric(df["session"], errors="coerce")
        # if everything NaN (unlikely), fallback to string
        if s.notna().sum() == 0:
            df["session"] = df["session"].astype(str)
        else:
            # nullable Int64 keeps NaN
            try:
                df["session"] = s.astype("Int64")
            except Exception:
                df["session"] = s  # float with NaN
    else:
        df["session"] = df["session"].astype(str)


def seq_length_series(df: pd.DataFrame) -> pd.Series:
    return df.groupby("sequence_id").size().rename("length")


def len_stats_by_class(df: pd.DataFrame) -> pd.DataFrame:
    # summarize per gesture_type based on per-sequence lengths
    def stats(g: pd.DataFrame) -> pd.Series:
        L = g.groupby("sequence_id").size()
        return pd.Series({
            "min": int(L.min()),
            "p25": float(L.quantile(0.25)),
            "median": float(L.median()),
            "p75": float(L.quantile(0.75)),
            "max": int(L.max()),
        })
    return df.groupby("gesture_type", sort=True).apply(stats, include_groups=False)


def coord_axis_ranges(df: pd.DataFrame, coord_cols: list[str]) -> pd.DataFrame:
    x_cols = [c for c in coord_cols if c.endswith("_x")]
    y_cols = [c for c in coord_cols if c.endswith("_y")]
    z_cols = [c for c in coord_cols if c.endswith("_z")]
    return pd.DataFrame({
        "axis": ["x", "y", "z"],
        "min": [df[x_cols].min().min(), df[y_cols].min().min(), df[z_cols].min().min()],
        "max": [df[x_cols].max().max(), df[y_cols].max().max(), df[z_cols].max().max()],
    })


def seq_velocity_norms(g: pd.DataFrame, coord_cols: list[str]) -> pd.Series:
    X = g[coord_cols].to_numpy(dtype=np.float64)
    if X.shape[0] < 2:
        return pd.Series({"v_mean": np.nan, "v_std": np.nan, "v_p95": np.nan, "v_max": np.nan})
    V = np.diff(X, axis=0)  # (T-1, 60)
    vnorm = np.linalg.norm(V, axis=1)
    return pd.Series({
        "v_mean": float(np.mean(vnorm)),
        "v_std": float(np.std(vnorm, ddof=1)) if vnorm.size > 1 else 0.0,
        "v_p95": float(np.percentile(vnorm, 95)),
        "v_max": float(np.max(vnorm)),
    })


def save_hist(data, bins, title, xlabel, ylabel, outpath: Path):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="EDA visualizations for Kinect skeleton dataset.")
    ap.add_argument("--input", type=str, required=True, help="Path to gesture_database.csv")
    ap.add_argument("--outdir", type=str, default="eda_out", help="Directory for outputs (figures, tables)")
    ap.add_argument("--per-class", action="store_true", help="Save histogram of sequence lengths per class")
    ap.add_argument("--per-performer", action="store_true", help="Save histogram of sequence lengths per performer")
    ap.add_argument("--max-classes", type=int, default=30, help="Limit number of class plots (avoid too many figures)")
    ap.add_argument("--max-performers", type=int, default=20, help="Limit number of performer plots")
    args = ap.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    # Low memory false to avoid dtype warning for wide CSV
    df = pd.read_csv(input_path, low_memory=False)

    # Detect coordinate columns
    coord_cols = detect_coord_columns(df)
    if len(coord_cols) != 60:
        raise ValueError(f"Expected 60 coordinate columns, got {len(coord_cols)}")

    # Normalize session dtype to avoid mixed-type sort issues
    coerce_session_dtype(df, prefer_numeric=True)

    # 1) Sequence lengths
    seq_lengths = seq_length_series(df)
    print("Sequence lengths summary:")
    print(seq_lengths.describe())

    # Save overall histogram
    save_hist(
        seq_lengths.values, bins=30,
        title="Sequence Lengths (All Sequences)",
        xlabel="Frames per sequence (T)",
        ylabel="Count of sequences",
        outpath=outdir / "hist_sequence_lengths_all.png"
    )

    # 1a) Per-class histograms (optional and potentially many)
    by_class_counts = df.groupby("gesture_type")["sequence_id"].nunique().sort_values(ascending=False)
    by_class_counts.to_csv(outdir / "sequences_per_class.csv", header=["n_sequences"])
    if args.per_class:
        top_classes = by_class_counts.index[:args.max_classes]
        class_dir = outdir / "hist_lengths_per_class"
        class_dir.mkdir(exist_ok=True)
        for cls in top_classes:
            L = df[df["gesture_type"] == cls].groupby("sequence_id").size()
            save_hist(
                L.values, bins=20,
                title=f"Sequence Lengths for Class: {cls}",
                xlabel="Frames per sequence (T)",
                ylabel="Count of sequences",
                outpath=class_dir / f"hist_lengths_{cls}.png"
            )

    # 1b) Per-performer histograms
    by_perf_counts = df.groupby("performer")["sequence_id"].nunique().sort_values(ascending=False)
    by_perf_counts.to_csv(outdir / "sequences_per_performer.csv", header=["n_sequences"])
    if args.per_performer:
        top_performers = by_perf_counts.index[:args.max_performers]
        perf_dir = outdir / "hist_lengths_per_performer"
        perf_dir.mkdir(exist_ok=True)
        for perf in top_performers:
            L = df[df["performer"] == perf].groupby("sequence_id").size()
            save_hist(
                L.values, bins=20,
                title=f"Sequence Lengths for Performer: {perf}",
                xlabel="Frames per sequence (T)",
                ylabel="Count of sequences",
                outpath=perf_dir / f"hist_lengths_{perf}.png"
            )

    # 1c) Length stats by class (table)
    length_stats = len_stats_by_class(df).sort_index()
    length_stats.to_csv(outdir / "length_stats_by_class.csv")
    print("Saved per-gesture length stats to:", outdir / "length_stats_by_class.csv")

    # 2) Coordinate ranges
    coord_ranges = coord_axis_ranges(df, coord_cols)
    coord_ranges.to_csv(outdir / "coordinate_axis_ranges.csv", index=False)
    print("Saved coordinate axis ranges to:", outdir / "coordinate_axis_ranges.csv")

    # 3) Outliers / jumps between frames via velocity norms
    dyn = (
        df.groupby("sequence_id", sort=False)
          .apply(lambda g: seq_velocity_norms(g, coord_cols), include_groups=False)
    )

    # Histogram of v_max
    save_hist(
        dyn["v_max"].dropna().values, bins=30,
        title="Distribution of Max Velocity Norm per Sequence",
        xlabel="v_max",
        ylabel="Count of sequences",
        outpath=outdir / "hist_vmax_per_sequence.png"
    )

    # IQR rule on v_max
    q1, q3 = dyn["v_max"].quantile(0.25), dyn["v_max"].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_mask = (dyn["v_max"] < low) | (dyn["v_max"] > high)
    outlier_pct = float(100 * outlier_mask.mean())

    top_outliers = dyn[outlier_mask].sort_values("v_max", ascending=False).head(50).reset_index()
    top_outliers.to_csv(outdir / "outlier_sequences_vmax_top50.csv", index=False)

    pd.DataFrame({
        "metric": ["v_max"],
        "q1": [q1],
        "q3": [q3],
        "iqr": [iqr],
        "low_thr": [low],
        "high_thr": [high],
        "outlier_pct": [outlier_pct],
    }).to_csv(outdir / "vmax_outlier_thresholds.csv", index=False)

    # Print concise summary
    print("\n=== Summary ===")
    print("Sequences per class (saved):", outdir / "sequences_per_class.csv")
    print("Sequences per performer (saved):", outdir / "sequences_per_performer.csv")
    print("Per-gesture length stats (saved):", outdir / "length_stats_by_class.csv")
    print("Coordinate ranges (saved):", outdir / "coordinate_axis_ranges.csv")
    print("v_max histogram (saved):", outdir / "hist_vmax_per_sequence.png")
    print("Outlier sequences (top 50 by v_max, saved):", outdir / "outlier_sequences_vmax_top50.csv")
    print("IQR thresholds (saved):", outdir / "vmax_outlier_thresholds.csv")
    print("Done.")


if __name__ == "__main__":
    main()
