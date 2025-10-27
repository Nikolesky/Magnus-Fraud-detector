# Magnus-Fraud-detector
A graph-based financial fraud detection system for bank transactions

# File structure:

Magnus-Fraud-detector/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 main.py
│
├── data/
│   ├── raw/                        ← 🧑‍💻 Kriti (Data Engineer)
│   │   └── transactions.csv
│   ├── processed/
│   │   ├── aggregated_features.csv ← 🧑‍💻 Kriti (output)
│   │   ├── node_features.csv       ← 🧑‍💻 Radhika (output)
│   │   └── risk_scores.csv         ← 🧑‍💻 Nikolesky (output)
│
├── models/
│   └── GraphFraudNet.pkl           ← 🧑‍💻 Nikolesky (ML Engineer)
│
├── results/
│   └── classification_report.txt   ← 🧑‍💻 Nikolesky (ML Engineer)
│
├── src/
│   ├── data_preprocess.py          ← 🧑‍💻 Kriti (Data Cleaning)
│   ├── graph_features.py           ← 🧑‍💻 Radhika (Graph Construction + Features)
│   ├── model_train.py              ← 🧑‍💻 Nikolesky (Model Training)
│   ├── risk_scorer.py              ← 🧑‍💻 Nikolesky (Risk Scoring)
│
└── ui/
   └── dashboard.py                ← 🧑‍💻 Shikavan (UI & Visualization)