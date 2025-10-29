#include <bits/stdc++.h>
using namespace std;

struct Transaction {
    string sender, receiver;
    double amount;
    int is_fraud;
};

struct AccountFeatures {
    int in_degree = 0, out_degree = 0;
    double sum_amount = 0.0;
    int tx_count = 0;
    double avg_neighbor_degree = 0.0;
    int is_fraud = 0;
    double pagerank = 0.0;
    double clustering = 0.0;
};

// Parse CSV file into transactions
vector<Transaction> parseCSV(const string& filename) {
    vector<Transaction> txs;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return txs;
    }
    string line;
    getline(file, line); // skip header
    
    while (getline(file, line)) {
        vector<string> fields;
        size_t start = 0, end = 0;
        //  split by comma
        while ((end = line.find(',', start)) != string::npos) {
            fields.push_back(line.substr(start, end - start));
            start = end + 1;
        }
        fields.push_back(line.substr(start));
        
        if (fields.size() < 6) {
            // malformed line, skip or handle error
            continue;
        }
        
        // Parse required fields
        string sender = fields[0];
        string receiver = fields[1];
        double amount = stod(fields[2]);
        int is_fraud = stoi(fields[5]);
        
        txs.push_back({sender, receiver, amount, is_fraud});
    }
    
    return txs;
}


//  assigning integer ID for account string
int getAccountId(const string& acc, unordered_map<string,int>& accToId, vector<string>& idToAcc) {
    auto it = accToId.find(acc);
    if (it != accToId.end())
        return it->second;
    int id = (int)idToAcc.size();
    accToId[acc] = id;
    idToAcc.push_back(acc);
    return id;
}

// To Build graph and account features using int IDs
void buildGraph(const vector<Transaction>& txs,
                vector<vector<int>>& adjList,
                vector<AccountFeatures>& feats,
                unordered_map<string,int>& accToId,
                vector<string>& idToAcc)
{
    for (const auto& tx : txs) {
        int u = getAccountId(tx.sender, accToId, idToAcc);
        int v = getAccountId(tx.receiver, accToId, idToAcc);
        
        // Ensure graph can hold all nodes
        if ((int)adjList.size() <= u) {
            adjList.resize(u+1);
            feats.resize(u+1);
        }
        if ((int)adjList.size() <= v) {
            adjList.resize(v+1);
            feats.resize(v+1);
        }
        
        adjList[u].push_back(v);
        
        feats[u].out_degree++;
        feats[u].sum_amount += tx.amount;
        feats[u].tx_count++;
        feats[v].in_degree++;
        
        feats[u].is_fraud |= tx.is_fraud;
        feats[v].is_fraud |= tx.is_fraud;
    }
}

// For calculating average neighbor degree per node
void computeNeighborDegree(const vector<vector<int>>& adjList, vector<AccountFeatures>& feats) {
    int N = (int)feats.size();
    for (int u = 0; u < N; ++u) {
        const auto& neighbors = adjList[u];
        if (neighbors.empty()) {
            feats[u].avg_neighbor_degree = 0.0;
            continue;
        }
        double sum_deg = 0.0;
        for (int nbr : neighbors) {
            sum_deg += feats[nbr].in_degree + feats[nbr].out_degree;
        }
        feats[u].avg_neighbor_degree = sum_deg / neighbors.size();
    }
}

//  For computing PageRank for graph
void computePageRank(const vector<vector<int>>& adjList, vector<double>& pagerank, 
                     double damping=0.85, double tol=1e-6, int max_iter=100) {
    int N = (int)adjList.size();
    pagerank.assign(N, 1.0 / N);

    vector<double> new_rank(N, 0.0);
    for (int iter = 0; iter < max_iter; ++iter) {
        double dangling_sum = 0.0;
        // Calculate the total rank of dangling nodes
        for (int u = 0; u < N; ++u) {
            if (adjList[u].empty()) dangling_sum += pagerank[u];
        }

        for (int u = 0; u < N; ++u) {
            new_rank[u] = (1.0 - damping) / N;
            new_rank[u] += damping * dangling_sum / N;
        }

        for (int u = 0; u < N; ++u) {
            int out_deg = adjList[u].size();
            if (out_deg == 0) continue;
            double share = damping * pagerank[u] / out_deg;
            for (int v : adjList[u]) {
                new_rank[v] += share;
            }
        }

        // Check if ranks have converged
        double err = 0.0;
        for (int u = 0; u < N; ++u) err += fabs(new_rank[u] - pagerank[u]);

        pagerank.swap(new_rank);
        if (err < tol) break; // Convergence achieved
        fill(new_rank.begin(), new_rank.end(), 0.0);
    }
}


// Compute clustering coefficient for each node
void computeClustering(const vector<vector<int>>& adjList, vector<AccountFeatures>& feats) {
    int N = (int)adjList.size();
    vector<unordered_set<int>> undirected(N);

    // Build undirected graph (adjacency sets)
    for (int u = 0; u < N; ++u) {
        for (int v : adjList[u]) {
            undirected[u].insert(v);
            undirected[v].insert(u);
        }
    }

    for (int u = 0; u < N; ++u) {
        int k = (int)undirected[u].size();
        if (k < 2) {
            feats[u].clustering = 0.0;
            continue;
        }

        int links = 0;
        vector<int> nbrVec(undirected[u].begin(), undirected[u].end());
        for (size_t i = 0; i < nbrVec.size(); ++i) {
            for (size_t j = i + 1; j < nbrVec.size(); ++j) {
                if (undirected[nbrVec[i]].count(nbrVec[j])) {
                    links++;
                }
            }
        }
        feats[u].clustering = (2.0 * links) / (k * (k - 1));
    }
}

// Write features to output CSV
void writeOutput(const string& filename, const vector<string>& idToAcc, const vector<AccountFeatures>& feats) {
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Error: cannot open output file " << filename << endl;
        return;
    }
    out << "account_id,in_degree,out_degree,avg_neighbor_degree,pagerank,clustering,is_fraud\n";
    int N = (int)feats.size();
    for (int i = 0; i < N; ++i) {
        out << idToAcc[i] << ","
            << feats[i].in_degree << ","
            << feats[i].out_degree << ","
            << feats[i].avg_neighbor_degree << ","
            << feats[i].pagerank << ","
            << feats[i].clustering << ","
            << feats[i].is_fraud << "\n";
    }
    out.close();
}

int main() {
    string filename = "../data/synthetic_transactions.csv";

    vector<Transaction> txs = parseCSV(filename);

    unordered_map<string, int> accToId;
    vector<string> idToAcc;
    vector<vector<int>> adjList;
    vector<AccountFeatures> feats;

    buildGraph(txs, adjList, feats, accToId, idToAcc);

    computeNeighborDegree(adjList, feats);

    vector<double> pagerank;
    computePageRank(adjList, pagerank);
    int N = (int)feats.size();
    for (int i = 0; i < N; ++i) {
        feats[i].pagerank = pagerank[i];
    }

    computeClustering(adjList, feats);

    writeOutput("../data/node_features.csv", idToAcc, feats);

    cout << "Done. Output written to node_features.csv\n";
    return 0;
}
