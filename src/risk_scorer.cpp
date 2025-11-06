#include <bits/stdc++.h>
using namespace std;

struct Transaction {
    string sender, receiver;
    double amount;
};

struct AccountFeatures {
    int in_degree = 0, out_degree = 0;
    double sum_amount = 0.0;
    int tx_count = 0;
    double avg_neighbor_degree = 0.0;
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

        // split by comma
        while ((end = line.find(',', start)) != string::npos) {
            fields.push_back(line.substr(start, end - start));
            start = end + 1;
        }
        fields.push_back(line.substr(start));
        
        if (fields.size() < 3) continue; // only expect sender, receiver, amount
        
        string sender = fields[0];
        string receiver = fields[1];
        double amount = stod(fields[2]);
        
        txs.push_back({sender, receiver, amount});
    }
    
    return txs;
}

// Assign integer ID for account strings
int getAccountId(const string& acc, unordered_map<string,int>& accToId, vector<string>& idToAcc) {
    auto it = accToId.find(acc);
    if (it != accToId.end())
        return it->second;
    int id = (int)idToAcc.size();
    accToId[acc] = id;
    idToAcc.push_back(acc);
    return id;
}

// Build graph and account features
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
        if ((int)adjList.size() <= max(u, v)) {
            adjList.resize(max(u, v) + 1);
            feats.resize(max(u, v) + 1);
        }
        
        adjList[u].push_back(v);
        
        feats[u].out_degree++;
        feats[u].sum_amount += tx.amount;
        feats[u].tx_count++;
        feats[v].in_degree++;
    }
}

// Compute average neighbor degree
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

// Compute PageRank
void computePageRank(const vector<vector<int>>& adjList, vector<double>& pagerank, 
                     double damping=0.85, double tol=1e-6, int max_iter=100) {
    int N = (int)adjList.size();
    if (N == 0) return;

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

        // Check convergence
        double err = 0.0;
        for (int u = 0; u < N; ++u) err += fabs(new_rank[u] - pagerank[u]);

        pagerank.swap(new_rank);
        if (err < tol) break;
        fill(new_rank.begin(), new_rank.end(), 0.0);
    }
}

// Compute clustering coefficient
void computeClustering(const vector<vector<int>>& adjList, vector<AccountFeatures>& feats) {
    int N = (int)adjList.size();
    vector<unordered_set<int>> undirected(N);

    // Build undirected graph
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

// Write features to CSV
void writeOutput(const string& filename, const vector<string>& idToAcc, const vector<AccountFeatures>& feats) {
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Error: cannot open output file " << filename << endl;
        return;
    }
    out << "account_id,in_degree,out_degree,avg_neighbor_degree,pagerank,clustering\n";
    int N = (int)feats.size();
    for (int i = 0; i < N; ++i) {
        out << idToAcc[i] << ","
            << feats[i].in_degree << ","
            << feats[i].out_degree << ","
            << feats[i].avg_neighbor_degree << ","
            << feats[i].pagerank << ","
            << feats[i].clustering << "\n";
    }
    out.close();
}

struct NodeFeatures {
    int in_degree, out_degree;
    double avg_neighbor_degree;
    int is_fraud; // (not used for prediction)
    double pagerank;
    double clustering;
};

struct Node {
    vector<double> weights;
    double bias = 0;
    double z;
    double output;
};

struct Layer {
    vector<Node> nodes;
};

struct NeuralNetwork {
    vector<Layer> network;
};

// === LOAD MODEL ===
NeuralNetwork load_model(const string &filename) {
    NeuralNetwork nn;

    ifstream in(filename, ios::binary);
    if (!in.is_open()) {
        cerr << "Error: Cannot open file " << filename << " for reading." << endl;
        return nn;
    }

    size_t num_layers;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    nn.network.resize(num_layers);

    for (auto &layer : nn.network) {
        size_t num_nodes;
        in.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
        layer.nodes.resize(num_nodes);

        for (auto &node : layer.nodes) {
            in.read(reinterpret_cast<char*>(&node.bias), sizeof(node.bias));
            size_t num_weights;
            in.read(reinterpret_cast<char*>(&num_weights), sizeof(num_weights));
            node.weights.resize(num_weights);
            if (num_weights > 0) {
                in.read(reinterpret_cast<char*>(node.weights.data()), num_weights * sizeof(double));
            }
        }
    }

    in.close();
    return nn;
}

// === PREDICT FUNCTION ===
double predict(NeuralNetwork &nn, const NodeFeatures &nf) {
    vector<double> inputs = {
        (double)nf.in_degree,
        (double)nf.out_degree,
        nf.avg_neighbor_degree,
        nf.pagerank,
        nf.clustering
    };

    for (int i = 0; i < nn.network[0].nodes.size() && i < inputs.size(); i++) {
        nn.network[0].nodes[i].output = inputs[i];
    }

    for (int i = 1; i < nn.network.size(); i++) {
        int prev_size = nn.network[i - 1].nodes.size();
        for (auto &node : nn.network[i].nodes) {
            double z = node.bias;
            for (int j = 0; j < prev_size; j++) {
                z += nn.network[i - 1].nodes[j].output * node.weights[j];
            }
            node.z = z;
            node.output = 1.0 / (1.0 + exp(-z));
        }
    }

    return nn.network.back().nodes[0].output;
}

// === MAIN ===
int main() {
    string filename;
    cout << "Enter path to transactions CSV file: ";
    cin >> filename;

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
    writeOutput("../results/node_input.csv", idToAcc, feats);

    cout << "Done. Output written to node_input.csv\n";

    NeuralNetwork nn = load_model("../models/modelx_1.pkl");  // <-- MISSING SEMICOLON FIXED

    ifstream file("../results/node_input.csv");  // <-- ADDED to replace missing file object
    ofstream out("../results/risk_scores.csv");
    out << "user_id,risk_score\n";

    string line;
    getline(file, line); // skip header
    while (getline(file, line)) {
        vector<string> fields;
        size_t start = 0, end = 0;
        while ((end = line.find(',', start)) != string::npos) {
            fields.push_back(line.substr(start, end - start));
            start = end + 1;
        }
        fields.push_back(line.substr(start));

        if (fields.size() < 6) continue;  // node_input.csv has 6 fields

        string user_id = fields[0];
        NodeFeatures nf;
        nf.in_degree = stoi(fields[1]);
        nf.out_degree = stoi(fields[2]);
        nf.avg_neighbor_degree = stod(fields[3]);
        nf.pagerank = stod(fields[4]);
        nf.clustering = stod(fields[5]);
        nf.is_fraud = 0;

        double risk = predict(nn, nf);
        out << user_id << "," << risk << "\n";
    }

    file.close();
    out.close();
    cout << "Predictions saved to ../results/risk_scores.csv" << endl;

    return 0;
}
