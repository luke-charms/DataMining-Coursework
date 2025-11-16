#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EDA for Kinect skeleton gesture dataset (no visualization).

- Expects a long/tidy table where each row = one frame.
- 60 coordinate columns (20 joints x,y,z) + metadata:
  sequence_id, performer, gesture_type, session
- Computes:
  * completeness & schema checks
  * sequence-length stats
  * coordinate ranges
  * counts by class / performer / session
  * finite-difference dynamics (velocity/acceleration) per sequence
  * robust outlier flags on dynamics
  * (optional) normalization diagnostics (hip-centered, shoulder-width scaled)

Usage:
    python EDA.py --input gesture_database.csv --outdir eda_out

Notes:
  - No plots. Outputs printed to console and (optionally) saved to CSV.
  - Handles mixed dtype in 'session' (string vs int) safely.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def detect_coord_columns(df: pd.DataFrame) -> list[str]:
    meta = {"sequence_id", "performer", "gesture_type", "session"}
    cols = [c for c in df.columns if c not in meta]
    return cols


def coerce_session_dtype(df: pd.DataFrame, prefer_numeric: bool = True) -> pd.DataFrame:
    """Normalize 'session' dtype to avoid mixed-type sorting errors."""
    if "session" not in df.columns:
        return df
    if prefer_numeric:
        s = pd.to_numeric(df["session"], errors="coerce")
        # If everything became NaN, fallback to string
        if s.notna().sum() == 0:
            df["session"] = df["session"].astype(str)
        else:
            # Use pandas nullable integer if possible
            try:
                df["session"] = s.astype("Int64")
            except Exception:
                df["session"] = s  # float with NaN
    else:
        df["session"] = df["session"].astype(str)
    return df


def schema_checks(df: pd.DataFrame, coord_cols: list[str]) -> dict:
    assert len(coord_cols) == 60, f"Expected 60 coordinate columns, found {len(coord_cols)}"
    required = ["sequence_id", "performer", "gesture_type", "session"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Missing and zero checks
    missing_any = df[coord_cols + required].isna().sum().sum()
    zeros_count = (df[coord_cols] == 0.0).sum().sum()

    # Basic counts
    info = {
        "total_sequences": int(df["sequence_id"].nunique()),
        "total_frames": int(len(df)),
        "classes": int(df["gesture_type"].nunique()),
        "performers": int(df["performer"].nunique()),
        "sessions": int(df["session"].nunique()),
        "missing_values_total": int(missing_any),
        "zero_values_total_coords_only": int(zeros_count),
    }
    return info


def sequence_length_stats(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    lengths = df.groupby("sequence_id").size()
    summary = {
        "min_len": int(lengths.min()),
        "p25_len": float(lengths.quantile(0.25)),
        "median_len": float(lengths.median()),
        "mean_len": float(lengths.mean()),
        "p75_len": float(lengths.quantile(0.75)),
        "max_len": int(lengths.max()),
    }
    return lengths, summary


def coordinate_ranges(df: pd.DataFrame, coord_cols: list[str]) -> dict:
    x_cols = [c for c in coord_cols if c.endswith("_x")]
    y_cols = [c for c in coord_cols if c.endswith("_y")]
    z_cols = [c for c in coord_cols if c.endswith("_z")]
    return {
        "x_min": float(df[x_cols].min().min()), "x_max": float(df[x_cols].max().max()),
        "y_min": float(df[y_cols].min().min()), "y_max": float(df[y_cols].max().max()),
        "z_min": float(df[z_cols].min().min()), "z_max": float(df[z_cols].max().max()),
    }


def counts_by(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    by_class = df.groupby("gesture_type")["sequence_id"].nunique().sort_values(ascending=False)
    by_perf = df.groupby("performer")["sequence_id"].nunique().sort_values(ascending=False)
    # session is normalized in dtype earlier; sort_index will work
    by_sess = df.groupby("session")["sequence_id"].nunique().sort_index()
    return by_class, by_perf, by_sess


def summarize_seq_dynamics(g: pd.DataFrame, coord_cols: list[str]) -> pd.Series:
    """
    For one sequence (group), compute velocity/acceleration norms and return summary stats.
    """
    X = g[coord_cols].to_numpy(dtype=np.float64)  # (T, 60)
    T = X.shape[0]

    # Velocity v_t = x_{t+1} - x_t  -> shape (T-1, 60)
    V = np.diff(X, axis=0) if T >= 2 else np.empty((0, X.shape[1]))
    # Acceleration a_t = x_{t+1} - 2x_t + x_{t-1} -> shape (T-2, 60)
    A = np.diff(X, n=2, axis=0) if T >= 3 else np.empty((0, X.shape[1]))

    v_norm = np.linalg.norm(V, axis=1) if V.shape[0] else np.array([])
    a_norm = np.linalg.norm(A, axis=1) if A.shape[0] else np.array([])

    def stats(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {"mean": np.nan, "std": np.nan, "p95": np.nan, "max": np.nan}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    vs = stats(v_norm)
    acc = stats(a_norm)
    return pd.Series({
        "v_mean": vs["mean"], "v_std": vs["std"], "v_p95": vs["p95"], "v_max": vs["max"],
        "a_mean": acc["mean"], "a_std": acc["std"], "a_p95": acc["p95"], "a_max": acc["max"],
    })


def robust_outlier_flags(dyn: pd.DataFrame, k: float = 1.5) -> dict:
    def iqr_mask(s: pd.Series, k: float) -> pd.Series:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        return (s < low) | (s > high)

    flags_vmax = iqr_mask(dyn["v_max"], k)
    flags_amax = iqr_mask(dyn["a_max"], k)

    return {
        "v_max_outlier_pct": float(100 * flags_vmax.mean()),
        "a_max_outlier_pct": float(100 * flags_amax.mean()),
    }


def normalization_diagnostics(df: pd.DataFrame, coord_cols: list[str]) -> pd.DataFrame:
    """
    Hip-center each frame and scale by shoulder width per frame, then summarize per sequence.
    Returns a long summary (feature × stat), can be heavy. Keep optional.
    """
    hip_x, hip_y, hip_z = "hip_center_x", "hip_center_y", "hip_center_z"
    ls_x, rs_x = "shoulder_left_x", "shoulder_right_x"
    ls_y, rs_y = "shoulder_left_y", "shoulder_right_y"
    ls_z, rs_z = "shoulder_left_z", "shoulder_right_z"

    def center_scale_group(g: pd.DataFrame) -> pd.Series:
        G = g.copy()

        # center by hip center
        for axis, hip in zip(("_x", "_y", "_z"), (hip_x, hip_y, hip_z)):
            cols = [c for c in coord_cols if c.endswith(axis)]
            G[cols] = G[cols].sub(G[hip], axis=0)

        # scale by shoulder width (per-frame)
        sw = np.sqrt(
            (G[ls_x] - G[rs_x]) ** 2 + (G[ls_y] - G[rs_y]) ** 2 + (G[ls_z] - G[rs_z]) ** 2
        )
        # Avoid division by zero: replace zeros with sequence median
        sw = sw.replace(0, np.nan)
        sw = sw.fillna(sw.median())

        for axis in ("_x", "_y", "_z"):
            cols = [c for c in coord_cols if c.endswith(axis)]
            G[cols] = G[cols].div(sw, axis=0)

        # summarize per-sequence after normalization
        desc = G[coord_cols].agg(["mean", "std", "min", "max"]).stack().rename("value")
        return desc

    norm_summary = df.groupby("sequence_id", sort=False).apply(center_scale_group)
    return norm_summary  # MultiIndex: (sequence_id, feature, stat) -> value


def save_series(s: pd.Series, path: Path, index_name: str, value_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = s.rename(value_name)
    s.index.name = index_name
    s.to_csv(path)


def save_frame(f: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(path, index=True)


def main():
    parser = argparse.ArgumentParser(description="EDA for Kinect skeleton gestures (no viz).")
    parser.add_argument("--input", type=str, required=True, help="Path to gesture_database.csv")
    parser.add_argument("--outdir", type=str, default="", help="Optional directory to save EDA outputs (CSV)")
    parser.add_argument("--normdiag", action="store_true", help="Compute heavy normalization diagnostics")
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else None

    print(f"Loading: {in_path}")
    df = pd.read_csv(in_path)

    # Normalize 'session' dtype to avoid TypeError on sort
    df = coerce_session_dtype(df, prefer_numeric=True)

    coord_cols = detect_coord_columns(df)
    print(f"Detected {len(coord_cols)} coordinate columns.")

    # Schema & completeness
    info = schema_checks(df, coord_cols)
    print("\n=== Completeness & Schema ===")
    for k, v in info.items():
        print(f"{k}: {v}")

    # Sequence length stats
    lengths, len_summary = sequence_length_stats(df)
    print("\n=== Sequence Length Summary ===")
    for k, v in len_summary.items():
        print(f"{k}: {v}")

    # Coordinate ranges
    cr = coordinate_ranges(df, coord_cols)
    print("\n=== Coordinate Ranges (overall) ===")
    for k, v in cr.items():
        print(f"{k}: {v}")

    # Counts by class/performer/session
    by_class, by_perf, by_sess = counts_by(df)
    print("\n=== Sequences per Class (unique sequences) ===")
    print(by_class)
    print("\n=== Sequences per Performer (unique sequences) ===")
    print(by_perf)
    print("\n=== Sequences per Session (unique sequences) ===")
    print(by_sess)

    # Dynamics per sequence
    print("\nComputing dynamics (velocity/acceleration) per sequence...")
    dyn = df.groupby("sequence_id", sort=False).apply(lambda g: summarize_seq_dynamics(g, coord_cols))
    print("\n=== Dynamics summary (per-sequence, describe) ===")
    print(dyn.describe().T)

    # Outlier flags
    outliers = robust_outlier_flags(dyn, k=1.5)
    print("\n=== Outlier summary (IQR rule, v_max & a_max) ===")
    for k, v in outliers.items():
        print(f"{k}: {v:.2f}%")

    # Optional normalization diagnostics (can be large)
    if args.normdiag:
        print("\nComputing normalization diagnostics (hip-centered, shoulder-width scaled)...")
        norm_summary = normalization_diagnostics(df, coord_cols)
        print("Normalization diagnostics computed (long table).")
    else:
        norm_summary = None

    # Save outputs
    if outdir:
        print(f"\nSaving outputs to: {outdir.resolve()}")
        save_series(lengths, outdir / "sequence_lengths.csv", "sequence_id", "length")
        save_frame(pd.DataFrame(len_summary, index=[0]), outdir / "sequence_length_summary.csv")
        save_frame(pd.DataFrame([cr]), outdir / "coordinate_ranges.csv")
        save_series(by_class, outdir / "sequences_per_class.csv", "gesture_type", "n_sequences")
        save_series(by_perf, outdir / "sequences_per_performer.csv", "performer", "n_sequences")
        save_series(by_sess, outdir / "sequences_per_session.csv", "session", "n_sequences")
        save_frame(dyn, outdir / "sequence_dynamics.csv")
        save_frame(pd.DataFrame([outliers]), outdir / "outlier_summary.csv")
        if norm_summary is not None:
            # MultiIndex -> tidy
            tidy = norm_summary.reset_index().rename(columns={"level_1": "feature", "level_2": "stat"})
            save_frame(tidy, outdir / "normalization_diagnostics.csv")
        print("Done.")


if __name__ == "__main__":
    main()
