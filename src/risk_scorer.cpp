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

// Reads transaction data from a CSV file
vector<Transaction> parseCSV(const string& filename) {
    vector<Transaction> txs;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return txs;
    }

    string line;
    getline(file, line);
    while (getline(file, line)) {
        vector<string> fields;
        size_t start = 0, end = 0;

        while ((end = line.find(',', start)) != string::npos) {
            fields.push_back(line.substr(start, end - start));
            start = end + 1;
        }
        fields.push_back(line.substr(start));

        if (fields.size() < 3) continue;
        txs.push_back({fields[0], fields[1], stod(fields[2])});
    }
    return txs;
}

// Assigns numeric IDs to account names
int getAccountId(const string& acc, unordered_map<string,int>& accToId, vector<string>& idToAcc) {
    if (accToId.find(acc) != accToId.end())
        return accToId[acc];
    int id = (int)idToAcc.size();
    accToId[acc] = id;
    idToAcc.push_back(acc);
    return id;
}

// Builds adjacency list and base features
void buildGraph(const vector<Transaction>& txs,
                vector<vector<int>>& adjList,
                vector<AccountFeatures>& feats,
                unordered_map<string,int>& accToId,
                vector<string>& idToAcc)
{
    for (const auto& tx : txs) {
        int u = getAccountId(tx.sender, accToId, idToAcc);
        int v = getAccountId(tx.receiver, accToId, idToAcc);

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

// Calculates average neighbor degree for each node
void computeNeighborDegree(const vector<vector<int>>& adjList, vector<AccountFeatures>& feats) {
    int N = feats.size();
    for (int u = 0; u < N; ++u) {
        if (adjList[u].empty()) {
            feats[u].avg_neighbor_degree = 0.0;
            continue;
        }
        double sum_deg = 0.0;
        for (int nbr : adjList[u])
            sum_deg += feats[nbr].in_degree + feats[nbr].out_degree;
        feats[u].avg_neighbor_degree = sum_deg / adjList[u].size();
    }
}

// Computes PageRank scores for all nodes
void computePageRank(const vector<vector<int>>& adjList, vector<double>& pagerank,
                     double damping = 0.85, double tol = 1e-6, int max_iter = 100) {
    int N = adjList.size();
    if (N == 0) return;

    pagerank.assign(N, 1.0 / N);
    vector<double> new_rank(N, 0.0);

    for (int iter = 0; iter < max_iter; ++iter) {
        double dangling_sum = 0.0;
        for (int u = 0; u < N; ++u)
            if (adjList[u].empty()) dangling_sum += pagerank[u];

        for (int u = 0; u < N; ++u)
            new_rank[u] = (1.0 - damping) / N + damping * dangling_sum / N;

        for (int u = 0; u < N; ++u) {
            if (adjList[u].empty()) continue;
            double share = damping * pagerank[u] / adjList[u].size();
            for (int v : adjList[u])
                new_rank[v] += share;
        }

        double err = 0.0;
        for (int u = 0; u < N; ++u)
            err += fabs(new_rank[u] - pagerank[u]);

        pagerank.swap(new_rank);
        if (err < tol) break;
        fill(new_rank.begin(), new_rank.end(), 0.0);
    }
}

// Computes clustering coefficient for each node
void computeClustering(const vector<vector<int>>& adjList, vector<AccountFeatures>& feats) {
    int N = adjList.size();
    vector<unordered_set<int>> undirected(N);

    for (int u = 0; u < N; ++u) {
        for (int v : adjList[u]) {
            undirected[u].insert(v);
            undirected[v].insert(u);
        }
    }

    for (int u = 0; u < N; ++u) {
        int k = undirected[u].size();
        if (k < 2) {
            feats[u].clustering = 0.0;
            continue;
        }

        int links = 0;
        vector<int> nbrVec(undirected[u].begin(), undirected[u].end());
        for (size_t i = 0; i < nbrVec.size(); ++i)
            for (size_t j = i + 1; j < nbrVec.size(); ++j)
                if (undirected[nbrVec[i]].count(nbrVec[j])) links++;

        feats[u].clustering = (2.0 * links) / (k * (k - 1));
    }
}

struct Edge { int to; double weight; };
struct GraphNode { int id; double bias, z, output; vector<Edge> edges; };
struct Layer { vector<GraphNode> nodes; };
struct NeuralNetwork { vector<Layer> layers; };

// Loads a trained neural network model
NeuralNetwork load_model(const string &filename) {
    NeuralNetwork nn;
    ifstream in(filename, ios::binary);
    if (!in.is_open()) {
        cerr << "Error: Cannot open model file " << filename << endl;
        return nn;
    }

    size_t num_layers;
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    nn.layers.resize(num_layers);

    int global_id = 0;
    for (auto &layer : nn.layers) {
        size_t num_nodes;
        in.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
        layer.nodes.resize(num_nodes);

        for (auto &node : layer.nodes) {
            node.id = global_id++;
            in.read(reinterpret_cast<char*>(&node.bias), sizeof(node.bias));

            size_t num_edges;
            in.read(reinterpret_cast<char*>(&num_edges), sizeof(num_edges));
            node.edges.resize(num_edges);

            for (auto &edge : node.edges) {
                in.read(reinterpret_cast<char*>(&edge.to), sizeof(edge.to));
                in.read(reinterpret_cast<char*>(&edge.weight), sizeof(edge.weight));
            }
        }
    }

    in.close();
    return nn;
}

// Finds a node by its ID in the neural network
GraphNode* findNodeById(vector<Layer> &layers, int id) {
    for (auto &L : layers)
        for (auto &n : L.nodes)
            if (n.id == id) return &n;
    return nullptr;
}

// Normalizes raw input features
vector<double> normalizeInputs(const vector<double>& inputs) {
    vector<double> scaled = inputs;
    vector<double> mins = {0, 0, 0, 0.000001, 0};
    vector<double> maxs = {1000, 1000, 1000, 0.01, 1};

    for (int i = 0; i < inputs.size(); i++) {
        scaled[i] = (inputs[i] - mins[i]) / (maxs[i] - mins[i] + 1e-9);
        scaled[i] = min(1.0, max(0.0, scaled[i]));
    }
    return scaled;
}

// Performs forward pass prediction using the trained model
double predict(NeuralNetwork &nn, const vector<double> &raw_inputs) {
    vector<double> inputs = normalizeInputs(raw_inputs);

    for (int i = 0; i < nn.layers[0].nodes.size() && i < inputs.size(); i++)
        nn.layers[0].nodes[i].output = inputs[i];

    for (int i = 1; i < nn.layers.size(); i++) {
        for (auto &node : nn.layers[i].nodes) {
            double z = node.bias;
            for (auto &edge : node.edges) {
                GraphNode *prev = findNodeById(nn.layers, edge.to);
                if (prev) z += prev->output * edge.weight;
            }
            node.z = z;
            node.output = 1.0 / (1.0 + exp(-z));
        }
    }

    return nn.layers.back().nodes.empty() ? 0.0 : nn.layers.back().nodes[0].output;
}

// Builds graph, extracts features, loads model, predicts fraud risk
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

    for (int i = 0; i < feats.size(); ++i)
        feats[i].pagerank = pagerank[i];

    computeClustering(adjList, feats);
    cout << "Graph features computed \n";

    NeuralNetwork nn = load_model("../models/model_graph.pkl");
    cout << "Graph-based model loaded \n";

    ofstream out("../results/risk_scores.csv");
    out << "user_id,risk_score\n";

    for (int i = 0; i < feats.size(); ++i) {
        vector<double> inputs = {
            (double)feats[i].in_degree,
            (double)feats[i].out_degree,
            feats[i].avg_neighbor_degree,
            feats[i].pagerank,
            feats[i].clustering
        };
        double risk = predict(nn, inputs);
        out << idToAcc[i] << "," << risk << "\n";
    }

    out.close();
    cout << "Predictions saved to ../results/risk_scores.csv\n";
    return 0;
}
