# EDA_3.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
CSV = "gesture_database.csv"  
df = pd.read_csv(CSV, low_memory=False)

# general statistics for sequence lengths
lengths = df.groupby("sequence_id").size().rename("T")
summary = lengths.describe(percentiles=[0.5, 0.9, 0.95]).to_frame("value")


# lengths by classes
def per_class_len_stats(g):
    L = g.groupby("sequence_id").size()
    return pd.Series({
        "min": int(L.min()),
        "p25": float(L.quantile(0.25)),
        "median": float(L.median()),
        "p75": float(L.quantile(0.75)),
        "max": int(L.max()),
        "mean": float(L.mean())
    })

# per-class length statistics
len_stats_by_class = (
    df.groupby("gesture_type", sort=True)
      .apply(per_class_len_stats)
      .sort_values("median")
)

# Print results
print("\n[Per-class length stats] (sorted by median):")
print(len_stats_by_class.to_string())

# Save to CSV
len_stats_by_class.to_csv("len_stats_by_class.csv")
print("\nSaved: len_stats_by_class.csv")



# 1) One length per sequence
seq_lengths = df.groupby("sequence_id").size().rename("T")  # frames per sequence
# Map each sequence_id to its gesture_type (take the first in the group)
seq_labels = df.groupby("sequence_id")["gesture_type"].first()

seq_meta = pd.DataFrame({
    "T": seq_lengths,
    "gesture_type": seq_labels
}).reset_index(drop=False)

# 2) Build list of length arrays per class
#    and sort classes by median length (ascending) for nicer ordering
per_class_lengths = (
    seq_meta.groupby("gesture_type")["T"]
            .apply(list)
            .to_frame("lengths")
)

per_class_lengths["median"] = per_class_lengths["lengths"].apply(lambda L: float(np.median(L)))
per_class_lengths = per_class_lengths.sort_values("median", ascending=True)

classes = per_class_lengths.index.tolist()
data = per_class_lengths["lengths"].tolist()

# 3) Plot: boxplot with classes on Y, lengths on X; whiskers = full range
fig_h = max(8, 0.35 * len(classes))  # scale height with #classes
plt.figure(figsize=(10, fig_h))

bp = plt.boxplot(
    data,
    vert=False,          # horizontal
    whis=[0, 100],       # whiskers at min and max (full range)
    showfliers=False,    # outliers are inside min-max anyway
    widths=0.6
)

# Labels and ticks
plt.yticks(ticks=np.arange(1, len(classes)+1), labels=classes)
plt.xlabel("Frames per sequence (T)")
plt.title("Sequence Lengths by Class (min-max whiskers, box = IQR)")

# Optional light grid
plt.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)

plt.tight_layout()
plt.savefig("boxplot_lengths_by_class.png", dpi=150)
plt.show()

print("Saved figure: boxplot_lengths_by_class.png")


# split by axes
coord_cols = [c for c in df.columns if c not in {"sequence_id","performer","gesture_type","session"}]
x_cols = [c for c in coord_cols if c.endswith("_x")]
y_cols = [c for c in coord_cols if c.endswith("_y")]
z_cols = [c for c in coord_cols if c.endswith("_z")]

ranges = {
    "x_min": float(df[x_cols].min().min()), "x_max": float(df[x_cols].max().max()),
    "y_min": float(df[y_cols].min().min()), "y_max": float(df[y_cols].max().max()),
    "z_min": float(df[z_cols].min().min()), "z_max": float(df[z_cols].max().max())
}
print("\n[Coordinate ranges] overall:", ranges)

# quick check for spikes in individual features
feat_minmax = df[coord_cols].agg(["min","max"]).T.sort_index()
bad_hi = feat_minmax["max"] > (ranges["z_max"] * 1.05)  # example threshold
bad_lo = feat_minmax["min"] < (ranges["x_min"] * 1.05)
suspicious = feat_minmax[bad_hi | bad_lo]
if not suspicious.empty:
    print("\n[Coordinate ranges] suspicious features:\n", suspicious)
else:
    print("\n[Coordinate ranges] no suspicious features beyond global extrema.")
