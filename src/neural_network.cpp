#include <bits/stdc++.h>
using namespace std;

// Structure to store one data sample (a user's network features)
struct NodeFeatures {
    int in_degree, out_degree;
    double avg_neighbor_degree;
    int is_fraud;
    double pagerank;
    double clustering;
};

// Represents one weighted connection between two nodes
struct Edge {
    int to;
    double weight;
};

// Represents one neuron (node) in the neural network
struct GraphNode {
    int id;
    double bias = 0;
    double z = 0;
    double output = 0;
    double delta = 0;
    vector<Edge> edges;
};

// Represents a layer in the neural network
struct Layer {
    vector<GraphNode> nodes;
};

// Reads the dataset from CSV and stores features
vector<NodeFeatures> parseCSV(const string &filename) {
    vector<NodeFeatures> data;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
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
        if (fields.size() < 7) continue;

        NodeFeatures f;
        f.in_degree = stoi(fields[1]);
        f.out_degree = stoi(fields[2]);
        f.avg_neighbor_degree = stod(fields[3]);
        f.pagerank = stod(fields[4]);
        f.clustering = stod(fields[5]);
        f.is_fraud = stoi(fields[6]);
        data.push_back(f);
    }
    return data;
}

// Initializes weights with small random values
vector<double> initialize_weights(int n_inputs) {
    double limit = sqrt(2.0 / max(1, n_inputs));
    static thread_local mt19937 gen((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_real_distribution<> dist(-limit, limit);
    vector<double> w(n_inputs);
    for (auto &wi : w) wi = dist(gen);
    return w;
}

// Builds a fully connected graph-based neural network
struct NeuralNetwork {
    vector<Layer> layers;
    NeuralNetwork() {
        vector<int> layer_sizes = {5, 16, 8, 4, 2, 1};
        int node_id = 0;
        layers.resize(layer_sizes.size());

        for (int i = 0; i < layer_sizes.size(); i++) {
            for (int j = 0; j < layer_sizes[i]; j++) {
                GraphNode node;
                node.id = node_id++;
                node.bias = 0.0;
                layers[i].nodes.push_back(node);
            }
        }

        static thread_local mt19937 gen((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());
        for (int i = 1; i < layer_sizes.size(); i++) {
            int prev_size = layer_sizes[i - 1];
            double limit = sqrt(2.0 / max(1, prev_size));
            uniform_real_distribution<> dist(-limit, limit);
            for (auto &node : layers[i].nodes) {
                for (auto &prev_node : layers[i - 1].nodes) {
                    node.edges.push_back({prev_node.id, dist(gen)});
                }
            }
        }
    }
};

// Finds a node by its ID
GraphNode* findNodeById(vector<Layer> &layers, int id) {
    for (auto &L : layers)
        for (auto &n : L.nodes)
            if (n.id == id) return &n;
    return nullptr;
}

// Computes average of each feature
vector<double> computeFeatureMeans(const vector<NodeFeatures>& data) {
    int n = data.size();
    vector<double> mean(5, 0.0);
    for (const auto &d : data) {
        mean[0] += d.in_degree;
        mean[1] += d.out_degree;
        mean[2] += d.avg_neighbor_degree;
        mean[3] += d.pagerank;
        mean[4] += d.clustering;
    }
    for (int i = 0; i < 5; ++i) mean[i] /= n;
    return mean;
}

// Computes standard deviation of each feature
vector<double> computeFeatureStd(const vector<NodeFeatures>& data, const vector<double>& mean) {
    int n = data.size();
    vector<double> stdv(5, 0.0);
    for (const auto &d : data) {
        stdv[0] += pow(d.in_degree - mean[0], 2);
        stdv[1] += pow(d.out_degree - mean[1], 2);
        stdv[2] += pow(d.avg_neighbor_degree - mean[2], 2);
        stdv[3] += pow(d.pagerank - mean[3], 2);
        stdv[4] += pow(d.clustering - mean[4], 2);
    }
    for (int i = 0; i < 5; ++i) stdv[i] = sqrt(stdv[i] / n + 1e-9);
    return stdv;
}

// Normalizes one data sample
vector<double> normalizeSample(const NodeFeatures &d, const vector<double>& mean, const vector<double>& stdv) {
    vector<double> x(5);
    x[0] = (d.in_degree - mean[0]) / (stdv[0] + 1e-9);
    x[1] = (d.out_degree - mean[1]) / (stdv[1] + 1e-9);
    x[2] = (d.avg_neighbor_degree - mean[2]) / (stdv[2] + 1e-9);
    x[3] = (d.pagerank - mean[3]) / (stdv[3] + 1e-9);
    x[4] = (d.clustering - mean[4]) / (stdv[4] + 1e-9);
    return x;
}

// Sigmoid activation function
inline double sigmoid(double x) {
    if (x >= 0) {
        double z = exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        double z = exp(x);
        return z / (1.0 + z);
    }
}

// Derivative of sigmoid (computed from output)
inline double dSigmoid_from_output(double out) {
    return out * (1.0 - out);
}

// Trains the neural network on the dataset
void train_model(NeuralNetwork &nn, vector<NodeFeatures> &features, int epochs = 100, double lr = 0.0025) {
    if (features.empty()) return;

    vector<double> mean = computeFeatureMeans(features);
    vector<double> stdv  = computeFeatureStd(features, mean);

    int total = features.size();
    int pos = 0;
    for (auto &f : features) if (f.is_fraud == 1) pos++;
    int neg = total - pos;
    double pos_weight = (pos > 0) ? (double)neg / (double)pos : 1.0;
    pos_weight = min(pos_weight, 20.0);

    mt19937 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        shuffle(features.begin(), features.end(), rng);
        double loss_sum = 0.0;
        int TP = 0, FP = 0, TN = 0, FN = 0;

        for (auto &nf : features) {
            vector<double> inputs = normalizeSample(nf, mean, stdv);
            for (int i = 0; i < nn.layers[0].nodes.size(); ++i)
                nn.layers[0].nodes[i].output = (i < inputs.size()) ? inputs[i] : 0.0;

            for (int li = 1; li < nn.layers.size(); ++li) {
                for (auto &node : nn.layers[li].nodes) {
                    double z = node.bias;
                    for (auto &edge : node.edges) {
                        GraphNode *prev = findNodeById(nn.layers, edge.to);
                        if (prev) z += prev->output * edge.weight;
                    }
                    node.z = z;
                    node.output = sigmoid(z);
                }
            }

            double y = nf.is_fraud ? 1.0 : 0.0;
            double pred = nn.layers.back().nodes[0].output;
            pred = min(1.0 - 1e-9, max(1e-9, pred));

            double sample_loss = -(y * log(pred) + (1.0 - y) * log(1.0 - pred));
            loss_sum += sample_loss;

            double grad = (pred - y);
            if (y == 1.0) grad *= pos_weight;
            nn.layers.back().nodes[0].delta = grad;

            for (int li = nn.layers.size() - 2; li >= 1; --li) {
                for (auto &node : nn.layers[li].nodes) {
                    double err = 0.0;
                    for (auto &nextNode : nn.layers[li + 1].nodes)
                        for (auto &edge : nextNode.edges)
                            if (edge.to == node.id) err += edge.weight * nextNode.delta;
                    node.delta = err * dSigmoid_from_output(node.output);
                }
            }

            for (int li = 1; li < nn.layers.size(); ++li) {
                for (auto &node : nn.layers[li].nodes) {
                    for (auto &edge : node.edges) {
                        GraphNode *prev = findNodeById(nn.layers, edge.to);
                        if (!prev) continue;
                        double grad_w = node.delta * prev->output;
                        grad_w = max(-5.0, min(5.0, grad_w));
                        edge.weight -= lr * grad_w;
                    }
                    double grad_b = max(-5.0, min(5.0, node.delta));
                    node.bias -= lr * grad_b;
                }
            }

            double predicted = (pred >= 0.5) ? 1.0 : 0.0;
            if (predicted == 1 && y == 1) TP++;
            else if (predicted == 0 && y == 1) FN++;
            else if (predicted == 1 && y == 0) FP++;
            else TN++;
        }

        double avg_loss = loss_sum / max(1, (int)features.size());
        if (epoch % 10 == 0 || epoch == 1)
            cout << "Epoch " << epoch << " | Loss: " << avg_loss << endl;
    }
}

// Saves the trained model to a file
void save_model(const NeuralNetwork &nn, const string &filename) {
    ofstream out(filename, ios::binary);
    if (!out.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return;
    }

    size_t num_layers = nn.layers.size();
    out.write(reinterpret_cast<const char *>(&num_layers), sizeof(num_layers));

    for (const auto &layer : nn.layers) {
        size_t num_nodes = layer.nodes.size();
        out.write(reinterpret_cast<const char *>(&num_nodes), sizeof(num_nodes));
        for (const auto &node : layer.nodes) {
            out.write(reinterpret_cast<const char *>(&node.bias), sizeof(node.bias));
            size_t num_edges = node.edges.size();
            out.write(reinterpret_cast<const char *>(&num_edges), sizeof(num_edges));
            for (const auto &edge : node.edges) {
                out.write(reinterpret_cast<const char *>(&edge.to), sizeof(edge.to));
                out.write(reinterpret_cast<const char *>(&edge.weight), sizeof(edge.weight));
            }
        }
    }
    out.close();
    cout << "Model saved to: " << filename << endl;
}

// Main entry point: loads data, trains the model, saves it
int main() {
    NeuralNetwork nn;
    vector<NodeFeatures> data = parseCSV("../data/node_features.csv");

    if (data.empty()) {
        cerr << "No data found in node_features.csv\n";
        return 1;
    }

    train_model(nn, data, 100, 0.0025);
    save_model(nn, "../models/model_graph.pkl");
    cout << "Training complete!" << endl;
    return 0;
}
