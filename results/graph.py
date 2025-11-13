import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===================== LOAD DATA =====================
# Load the risk scores
df = pd.read_csv("risk_scores.csv")

# Ensure column names are lowercase and trimmed
df.columns = [c.strip().lower() for c in df.columns]

# Check if required columns exist
if "risk_score" not in df.columns:
    raise ValueError("CSV must contain a 'risk_score' column!")

# ===================== NORMALIZATION =====================
# Normalize risk scores to 0–1 range for meaningful visualization
min_score = df["risk_score"].min()
max_score = df["risk_score"].max()

if max_score != min_score:
    df["normalized_risk"] = (df["risk_score"] - min_score) / (max_score - min_score)
else:
    df["normalized_risk"] = 0.0  # fallback if all scores are identical

# ===================== PLOTTING =====================
plt.figure(figsize=(10, 5))
plt.plot(df["normalized_risk"], color='steelblue', linewidth=1.8, label="Normalized Risk")

# Highlight top 5% risky users
threshold = np.percentile(df["normalized_risk"], 95)
risky_users = df["normalized_risk"] >= threshold
plt.scatter(df.index[risky_users], df["normalized_risk"][risky_users],
            color='red', s=15, label="Top 5% Risky")

plt.title("Fraud Risk Scores per User", fontsize=14, weight='bold')
plt.xlabel("User Index", fontsize=12)
plt.ylabel("Normalized Risk Score", fontsize=12)
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# ===================== SAVE & SHOW =====================
plt.show()
 