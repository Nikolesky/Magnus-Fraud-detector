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
    double delta; //for backpropagation appaerently
};

struct Layer {
    vector<Node> nodes;
};

vector<NodeFeatures> parseCSV(const string& filename) {
    // Implementation of CSV parsing
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
        //  split by comma
        while ((end = line.find(',', start)) != string::npos) {
            fields.push_back(line.substr(start, end - start));
            start = end + 1;
        }
        fields.push_back(line.substr(start));

        if (fields.size() < 7) {
            // malformed line, skip or handle error
            continue;
        }

        // Parse required fields
        string account_id = fields[0];
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
    for (auto &wi : w)
        wi = dist(gen);

    return w;
}

struct NeuralNetwork {
    vector<Layer> network;

    NeuralNetwork() {
        network.resize(6);

        // Initialize network with random weights
        vector<double> w0 = {0}; //dummy weights for input layer
        vector<double> w1 = initialize_weights(5);
        vector<double> w2 = initialize_weights(16);
        vector<double> w3 = initialize_weights(8);
        vector<double> w4 = initialize_weights(4);
        vector<double> w5 = initialize_weights(2);

        network[0].nodes.resize(5);
        network[1].nodes.resize(16);
        network[2].nodes.resize(8);
        network[3].nodes.resize(4);
        network[4].nodes.resize(2);
        network[5].nodes.resize(1);

        // assign weights to first node in each layer for now
        network[0].nodes[0].weights = w0;
        network[1].nodes[0].weights = w1;
        network[2].nodes[0].weights = w2;
        network[3].nodes[0].weights = w3;
        network[4].nodes[0].weights = w4;
        network[5].nodes[0].weights = w5;
    }
};

void train_model(NeuralNetwork &nn, const NodeFeatures &nf) {
    // Forward propagation
    vector<double> inputs = {
        (double)nf.in_degree,
        (double)nf.out_degree,
        nf.avg_neighbor_degree,
        nf.pagerank,
        nf.clustering
    };

    // assign inputs to input layer outputs
    for (int i = 0; i < nn.network[0].nodes.size() && i < inputs.size(); i++) {
        nn.network[0].nodes[i].output = inputs[i];
    }

    // Forward pass for hidden and output layers
    for (int i = 1; i < nn.network.size(); i++) {
        int prev_size = nn.network[i - 1].nodes.size();
        for (auto &node : nn.network[i].nodes) {
            if (node.weights.empty()) {
                node.weights = initialize_weights(prev_size); // initialize if not set
            }

            double z = 0;
            for (int j = 0; j < prev_size; j++) {
                z += nn.network[i - 1].nodes[j].output * node.weights[j];
            }
            z += node.bias;
            node.z = z;
            node.output = 1.0 / (1.0 + exp(-z)); // sigmoid activation
        }
    }

    // Expected output (label)
    double y = (double)nf.is_fraud;
    double N = 0.000001; // learning rate

    // Compute output layer delta
    Layer &output_layer = nn.network.back();
    for (auto &node : output_layer.nodes) {
        double a = node.output;
        node.delta = (a - y) * a * (1 - a); // derivative of sigmoid * error
    }

    // Backpropagate errors to hidden layers
    for (int i = nn.network.size() - 2; i >= 1; i--) {
        Layer &current = nn.network[i];
        Layer &next = nn.network[i + 1];

        for (int j = 0; j < current.nodes.size(); j++) {
            double error_sum = 0;
            for (auto &next_node : next.nodes) {
                if (j < next_node.weights.size()) {
                    error_sum += next_node.weights[j] * next_node.delta;
                }
            }
            current.nodes[j].delta = error_sum * current.nodes[j].output * (1 - current.nodes[j].output);
        }
    }

    // Update weights and biases
    for (int i = 1; i < nn.network.size(); i++) {
        Layer &current = nn.network[i];
        Layer &previous = nn.network[i - 1];
        for (auto &node : current.nodes) {
            for (int j = 0; j < node.weights.size(); j++) {
                node.weights[j] -= N * node.delta * previous.nodes[j].output;
            }
            node.bias -= N * node.delta;
        }
    }
}


void save_model(const NeuralNetwork &nn, const string &filename) {
    ofstream out(filename, ios::binary);
    if (!out.is_open()) {
        cerr << "Error: Cannot open file " << filename << " for writing." << endl;
        return;
    }

    // Save number of layers
    size_t num_layers = nn.network.size();
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    // Save each layer
    for (const auto &layer : nn.network) {
        size_t num_nodes = layer.nodes.size();
        out.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));

        // Save each node
        for (const auto &node : layer.nodes) {
            // Save bias
            out.write(reinterpret_cast<const char*>(&node.bias), sizeof(node.bias));

            // Save weights
            size_t num_weights = node.weights.size();
            out.write(reinterpret_cast<const char*>(&num_weights), sizeof(num_weights));
            if (num_weights > 0) {
                out.write(reinterpret_cast<const char*>(node.weights.data()), num_weights * sizeof(double));
            }
        }
    }

    out.close();
}


int main() {
    NeuralNetwork nn;

    vector<NodeFeatures> features = parseCSV("../data/node_features.csv");

    //Give inputs to neural network
    for (auto nf : features) {
        train_model(nn, nf);
    }

    //save model weights
    save_model(nn, "../models/model_2.pkl");

    return 0;
}
