# EDA_2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the gesture database CSV file
CSV = "gesture_database.csv"  
df = pd.read_csv(CSV, low_memory=False)

# Analysis of sequence lengths
lengths = df.groupby("sequence_id").size().rename("T")
summary = lengths.describe(percentiles=[0.5, 0.9, 0.95]).to_frame("value")

# Calculate P90 and P95
P90 = float(lengths.quantile(0.90))
P95 = float(lengths.quantile(0.95))

# Display summary statistics
print("\n[Lengths] summary:\n", summary)
print(f"[Lengths] P90={P90:.1f}, P95={P95:.1f}")

# Plot histogram of sequence lengths
plt.figure()
plt.hist(lengths.values, bins=30)
plt.title("Sequence Lengths (All Sequences)")
plt.xlabel("Frames per sequence (T)")
plt.ylabel("Count of sequences")
plt.tight_layout()
plt.show()
