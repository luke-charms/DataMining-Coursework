import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Configuration

CSV = "gesture_database.csv"   # path to input CSV file with gesture data
MIN_RUN = 3                    # low-motion steps
FIXED_EPS = None               # set to a float to use a global threshold; None -> adaptive per sequence (q10)
PLOT = True


# Load data and identify features

df = pd.read_csv(CSV, low_memory=False)
meta_cols = {"sequence_id", "performer", "gesture_type", "session"}
coord_cols = [c for c in df.columns if c not in meta_cols]
assert len(coord_cols) == 60, f"Expected 60 coordinate columns, got {len(coord_cols)}"

# Helper functions

def per_sequence_velocities(g: pd.DataFrame) -> np.ndarray:
    """
    Returns the L2 velocity magnitudes v_t for a sequence g:
    v_t = ||x_{t+1} - x_t||_2, t=0..T-2
    """
    X = g[coord_cols].to_numpy(dtype=np.float64)
    if X.shape[0] < 2:
        return np.array([], dtype=np.float64)
    V = np.diff(X, axis=0)                 # (T-1, 60)
    vnorm = np.linalg.norm(V, axis=1)      # (T-1,)
    return vnorm

def silence_runs_for_seq(g: pd.DataFrame,
                         fixed_eps: float | None = None,
                         min_run: int = 3) -> pd.Series:
    """
    Counts leading/trailing low-motion runs using ||v_t|| < eps.
    If fixed_eps is None, choose eps adaptively as q10 of non-zero velocities in the sequence.
    Returns lead_silent, tail_silent, eps_used (and T, the original frame count).
    """
    T = g.shape[0]
    v = per_sequence_velocities(g)  # length T-1
    if T < 3 or v.size == 0:
        return pd.Series({"T": T, "lead_silent": 0, "tail_silent": 0, "eps_used": np.nan})

    if fixed_eps is None:
        nz = v[v > 0]
        eps_used = float(np.percentile(nz, 10)) if nz.size > 0 else 0.0
    else:
        eps_used = float(fixed_eps)

    # leading silence
    lead = 0
    for val in v:
        if val < eps_used: lead += 1
        else: break

    # trailing silence
    tail = 0
    for val in v[::-1]:
        if val < eps_used: tail += 1
        else: break

    # enforce minimum consecutive length
    if lead < min_run: lead = 0
    if tail < min_run: tail = 0

    return pd.Series({"T": T, "lead_silent": int(lead), "tail_silent": int(tail), "eps_used": eps_used})

def trimming_window(T: int, lead: int, tail: int) -> tuple[int, int]:
    """
    Map silence counts (measured on v_t) back to frame indices [start, end] inclusive.
    Dropping 'lead' diffs means start at frame index 'lead'.
    Dropping 'tail' diffs means end at frame index (T-1 - tail).
    Ensures at least one frame remains.
    """
    start = max(0, lead)
    end = T - 1 - max(0, tail)
    if end < start:
        end = start
    return start, end


# Compute silence stats per sequence

sil = df.groupby("sequence_id", sort=False).apply(
    lambda g: silence_runs_for_seq(g, fixed_eps=FIXED_EPS, min_run=MIN_RUN)
)

# Attach suggested trimming windows
lengths = df.groupby("sequence_id").size().rename("T_check")
res = pd.concat([sil, lengths], axis=1)
res["keep_start_end"] = res.apply(
    lambda r: trimming_window(int(r["T"]), int(r["lead_silent"]), int(r["tail_silent"])), axis=1
)


# Save artifacts

res.to_csv("silence_trimming_summary.csv", index=True)
print("Saved: silence_trimming_summary.csv")
print("\n[Silence summary] (frames):")
print(res[["T", "lead_silent", "tail_silent"]].describe().T)

# Show examples with the longest trailing silence (likely to benefit most)
print("\nExamples with longest trailing silence:")
print(res.sort_values("tail_silent", ascending=False).head(10))


# Optional plots

if PLOT:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,4))
    plt.hist(res["lead_silent"], bins=30)
    plt.title("Distribution of leading low-motion frames")
    plt.xlabel("lead_silent (frames)")
    plt.ylabel("Count of sequences")
    plt.tight_layout()
    plt.savefig("hist_lead_silence.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8,4))
    plt.hist(res["tail_silent"], bins=30)
    plt.title("Distribution of trailing low-motion frames")
    plt.xlabel("tail_silent (frames)")
    plt.ylabel("Count of sequences")
    plt.tight_layout()
    plt.savefig("hist_tail_silence.png", dpi=150)
    plt.close()

    print("Saved: hist_lead_silence.png, hist_tail_silence.png")

