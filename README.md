# Magnus-Fraud-detector
A graph-based financial fraud detection system for bank transactions

# Overview

This project implements a **Graph-Based Financial Fraud Detection System** entirely in **C++**, using fundamental **Data Structures and Algorithms** — without relying on any external ML or graph libraries.

It models bank transactions as a **directed weighted graph**, extracts structural and behavioral features for each account, and applies a **custom-trained mathematical fraud detection model** built from scratch.

The goal is to detect suspicious transaction patterns such as:
- **Cyclic money flows** (A → B → C → A)
- **Densely connected clusters** (potential fraud rings)
- **Highly central nodes** (accounts acting as money hubs)

# Input Format: 

**You can input .csv files in the following format:**
```
sender,receiver,amount,timestamp,type
U128,U28,2770.33,2025-10-21 12:20:59,cash_in
U10,U47,5630.12,2025-10-21 15:11:05,transfer
```

# Output format:
```
Account_ID  | Fraud_Score
U47         | 0.92
U85         | 0.78
U128        | 0.15
```

# File structure:

```
Magnus-Fraud-detector/
│
├── 📄 README.md
├── 📄 requirements.txt
│
├── data/
│   ├── synthetic_transactions.csv                  
│   ├── node_features.csv     
│   └── risk_scores.csv         
│
├── models/
│   ├── model_1.pk1
│   └── model_2.pkl           
│
├── results/
│   └── classification_report.txt   
│
├── src/         
│   ├── graph_features.cpp        
│   ├── graph_features.exe
│   ├── neural_network.cpp  
│   ├── neural_netowork.exe         
│   └── risk_scorer.cpp              
│
└── ui/
   └── dashboard.py

```