# EDA_4.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load gesture database (created earlier)
df = pd.read_csv("gesture_database.csv", low_memory=False)

# Identify coordinate columns
coord_cols = [c for c in df.columns if c not in {"sequence_id", "performer", "gesture_type", "session"}]

# Compute per-sequence velocity magnitudes
def sequence_velocity_stats(group):
    """
    Compute velocity (frame-to-frame Euclidean displacement)
    and summarize per sequence.
    """
    arr = group[coord_cols].to_numpy()
    # Frame-to-frame difference
    diff = np.diff(arr, axis=0)
    # Euclidean magnitude of velocity for each frame
    v = np.linalg.norm(diff, axis=1)
    
    return pd.Series({
        "v_mean": np.mean(v),
        "v_std": np.std(v),
        "v_max": np.max(v),
        "v_p95": np.percentile(v, 95)
    })

# Apply per-sequence
vel_stats = df.groupby("sequence_id", sort=False).apply(sequence_velocity_stats)

# Detect sequences with unusually high v_max (outliers)
# Using IQR rule
Q1 = vel_stats["v_max"].quantile(0.25)
Q3 = vel_stats["v_max"].quantile(0.75)
IQR = Q3 - Q1
upper_threshold = Q3 + 1.5 * IQR

outliers = vel_stats[vel_stats["v_max"] > upper_threshold]

print(f"Detected {len(outliers)} outlier sequences out of {len(vel_stats)} ({len(outliers)/len(vel_stats)*100:.2f}%).")
print("\nTop outlier examples:")
print(outliers.sort_values("v_max", ascending=False).head(10))

# Visualization: Boxplot of v_max

plt.figure(figsize=(8, 5))
plt.boxplot(vel_stats["v_max"], vert=False)
plt.title("Distribution of Maximum Velocity (v_max) per Sequence")
plt.xlabel("Velocity magnitude")
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("vmax_boxplot.png", dpi=150)
plt.show()

# Save results for inspection
vel_stats.to_csv("velocity_stats.csv", index=True)
outliers.to_csv("velocity_outliers.csv", index=True)
print("\nSaved: velocity_stats.csv and velocity_outliers.csv")
