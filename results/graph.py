import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("risk_scores.csv")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df["risk_score"], color='steelblue', linewidth=2)
plt.title("Fraud Risk Scores per User", fontsize=14)
plt.xlabel("User Index", fontsize=12)
plt.ylabel("Risk Score", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
