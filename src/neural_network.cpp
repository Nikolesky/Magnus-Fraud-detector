#include <bits/stdc++.h>
using namespace std;

struct NodeFeatures {
    int in_degree, out_degree;
    double avg_neighbor_degree;
    int is_fraud;
    double pagerank;
    double clustering;
};

struct Node {
    vector<double> weights;
    double bias = 0;
    double z;
    double output;
    double delta;
};

struct Layer {
    vector<Node> nodes;
};

vector<NodeFeatures> parseCSV(const string& filename) {
    vector<NodeFeatures> Nf;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return Nf;
    }

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

        if (fields.size() < 7) continue;

        int in_degree = stoi(fields[1]);
        int out_degree = stoi(fields[2]);
        double avg_neighbor_degree = stod(fields[3]);
        double pagerank = stod(fields[4]);
        double clustering = stod(fields[5]);
        int is_fraud = stoi(fields[6]);

        Nf.push_back({in_degree, out_degree, avg_neighbor_degree, is_fraud, pagerank, clustering});
    }

    return Nf;
}

vector<double> initialize_weights(int n_inputs) {
    double limit = sqrt(1.0 / n_inputs);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(-limit, limit);

    vector<double> w(n_inputs);
    for (auto &wi : w) wi = dist(gen);
    return w;
}

struct NeuralNetwork {
    vector<Layer> network;

    NeuralNetwork() {
        // layer sizes: 5 → 16 → 8 → 4 → 2 → 1
        vector<int> layer_sizes = {5, 16, 8, 4, 2, 1};
        network.resize(layer_sizes.size());

        for (int i = 0; i < layer_sizes.size(); i++) {
            network[i].nodes.resize(layer_sizes[i]);
            if (i > 0) {
                int prev_size = layer_sizes[i - 1];
                for (auto &node : network[i].nodes) {
                    node.weights = initialize_weights(prev_size);
                }
            }
        }
    }
};

void train_model(NeuralNetwork &nn, const NodeFeatures &nf) {
    vector<double> inputs = {
        (double)nf.in_degree,
        (double)nf.out_degree,
        nf.avg_neighbor_degree,
        nf.pagerank,
        nf.clustering
    };

    // Forward pass
    for (int i = 0; i < nn.network[0].nodes.size(); i++)
        nn.network[0].nodes[i].output = inputs[i];

    for (int i = 1; i < nn.network.size(); i++) {
        for (auto &node : nn.network[i].nodes) {
            double z = node.bias;
            for (int j = 0; j < nn.network[i - 1].nodes.size(); j++)
                z += nn.network[i - 1].nodes[j].output * node.weights[j];
            node.z = z;
            node.output = 1.0 / (1.0 + exp(-z)); // sigmoid
        }
    }

    double y = nf.is_fraud;
    double lr = 0.001;

    // Output delta
    Layer &output = nn.network.back();
    for (auto &node : output.nodes) {
        double a = node.output;
        node.delta = (a - y) * a * (1 - a);
    }

    // Hidden deltas
    for (int i = nn.network.size() - 2; i >= 1; i--) {
        Layer &L = nn.network[i];
        Layer &next = nn.network[i + 1];
        for (int j = 0; j < L.nodes.size(); j++) {
            double err = 0;
            for (auto &nxt : next.nodes)
                if (j < nxt.weights.size())
                    err += nxt.weights[j] * nxt.delta;
            L.nodes[j].delta = err * L.nodes[j].output * (1 - L.nodes[j].output);
        }
    }

    // Update weights
    for (int i = 1; i < nn.network.size(); i++) {
        Layer &L = nn.network[i];
        Layer &prev = nn.network[i - 1];
        for (auto &node : L.nodes) {
            for (int j = 0; j < node.weights.size(); j++)
                node.weights[j] -= lr * node.delta * prev.nodes[j].output;
            node.bias -= lr * node.delta;
        }
    }
}

void save_model(const NeuralNetwork &nn, const string &filename) {
    ofstream out(filename, ios::binary);
    if (!out.is_open()) {
        cerr << "Error: Cannot open file " << filename << " for writing." << endl;
        return;
    }

    size_t num_layers = nn.network.size();
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    for (const auto &layer : nn.network) {
        size_t num_nodes = layer.nodes.size();
        out.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));

        for (const auto &node : layer.nodes) {
            out.write(reinterpret_cast<const char*>(&node.bias), sizeof(node.bias));
            size_t num_weights = node.weights.size();
            out.write(reinterpret_cast<const char*>(&num_weights), sizeof(num_weights));
            if (num_weights > 0)
                out.write(reinterpret_cast<const char*>(node.weights.data()), num_weights * sizeof(double));
        }
    }

    out.close();
    cout << "Model saved to: " << filename << endl;
}

int main() {
    NeuralNetwork nn;
    vector<NodeFeatures> features = parseCSV("../data/node_features.csv");

    if (features.empty()) {
        cerr << "No data found in node_features.csv\n";
        return 1;
    }

    for (auto &nf : features)
        train_model(nn, nf);

    save_model(nn, "../models/modelx_1.pkl");
    cout << "Training complete ✅" << endl;

    return 0;
}
