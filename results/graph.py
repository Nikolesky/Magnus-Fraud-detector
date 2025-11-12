import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load csv file
df = pd.read_csv("risk_scores.csv")

# make column names lowercase
df.columns = [c.strip().lower() for c in df.columns]

# check if we have risk_score column
if "risk_score" not in df.columns:
    print("Error: risk_score column not found!")
    exit()

# normalize scores between 0 and 1
min_score = df["risk_score"].min()
max_score = df["risk_score"].max()

if max_score != min_score:
    df["normalized_risk"] = (df["risk_score"] - min_score) / (max_score - min_score)
else:
    df["normalized_risk"] = 0.0  # if all same values

# plot the scores
plt.figure(figsize=(10, 5))
plt.plot(df["normalized_risk"], color='blue', linewidth=2, label="risk")

# mark top 5 percent risky users
threshold = np.percentile(df["normalized_risk"], 95)
risky_users = df["normalized_risk"] >= threshold
plt.scatter(df.index[risky_users], df["normalized_risk"][risky_users],
            color='red', s=20, label="top 5%")

plt.title("Fraud Risk Scores", fontsize=14)
plt.xlabel("user index")
plt.ylabel("risk score")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()
