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
│   ├── model_graph1.pkl
│   └── model_graph2.pkl           
│
├── results/
│   ├── node_input.csv
│   ├── risk_scorer.csv
|   ├── network_metrics.csv
|   ├── suspicious_transactions.csv
|   ├── user_statistics
│   └── graph.py   
│
├── src/         
│   ├── model trainer
│   │    ├── graph_features.cpp
│   │    ├── graph_features.exe
│   │    ├── neural_network.cpp  
│   │    └── neural_netowork.exe        
│   │
|   ├── user_study.py
|   ├── generating_dataset.py
|   ├── calculatingusingdatastructs.cpp
|   ├── calculatingusingdatastructs.exe
│   ├── risk_scorer.cpp
│   └── risk_scorer.exe              
│
└── ui/
   └── dashboard.py

```
# CONTRIBUTION:

Kriti Agarwal: 
- Performed exploratory analysis of the transaction dataset, identified key fraud-related patterns, cleaned and prepared data for downstream modules, and generated insights that guided feature design.

Radhika Nijhara:
- Designed and implemented the transaction-graph architecture, mapped accounts and transfers into a graph structure, and built algorithms to extract structural fraud indicators such as hubs, cycles, and abnormal connectivity.

Vempati Nityan:
- Developed and trained the neural-network fraud detection model, optimized hyperparameters, integrated processed graph/data features, and produced final risk-score outputs for each account.

Vanshika Mehta
- Built the user-facing dashboard to visualise risk scores clearly, designed a clean interface for non-technical users, and ensured results were presented in an interpretable and reproducible manner.

# FLOW:

**Graph Feature Extraction (graph_features.cpp for training and risk_scorer.cpp for fraud scoring):**
```
This module processes the transaction data and creates a graph-based representation of the network.
Each account is treated as a node, and every transaction between two accounts is represented as a directed edge weighted by the transaction amount.

It then calculates several important features for each account, such as:

 -In-degree and Out-degree – number of incoming and outgoing transactions
 -Average Neighbor Degree – how connected an account’s neighbors are
 -PageRank – measures how central or important an account is in the network
 -Clustering Coefficient – checks how strongly an account’s neighbors are connected, which can help identify fraud rings

Finally, all the computed features are saved in node_features.csv, which is later used for fraud scoring and model training.

```

**Neural Network Architecture (neural_network.cpp for training and risk_scorer.cpp for implementing):**
```
This module processes the node features created during graph feature extraction. 
It creates a forward-pass neural network, normalizes and passes the graph metrics into the input layer.
The nodes of the input layer are connected to the nodes of successive layer through a graph data structure of graphnodes, where each graphnode stores a hashmap which consists of indexing and pointer torwards the specific node where all the information related to that node is stored.

-Input Layer: normalized graph metrics per account
-Hidden Layers: connected through graph-inspired edges
-Output Layer: fraud probability (risk score)
-Connections: stored as a graph (vector<edges> per node)
-Weights: stored using an unordered_map<int, GraphNode*> for O(1) access
```
