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

# ===================== PLOTTING =====================
plt.figure(figsize=(10, 5))
plt.plot(df["risk_score"], color='steelblue', linewidth=1.8, label="Risk Score")

# Highlight top 5% risky users
threshold = np.percentile(df["risk_score"], 95)
risky_users = df["risk_score"] >= threshold
plt.scatter(df.index[risky_users], df["risk_score"][risky_users],
            color='red', s=15, label="Top 5% Risky")

plt.title("Fraud Risk Scores per User", fontsize=14, weight='bold')
plt.xlabel("User Index", fontsize=12)
plt.ylabel("Risk Score", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# ===================== SAVE & SHOW =====================
plt.show()
