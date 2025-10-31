#include <bits/stdc++.h>;

struct AccountFeatures {
    int in_degree, out_degree;
    double avg_neighbor_degree;
    int is_fraud;
    double pagerank;
    double clustering;
};

struct Node {
    vector<double> weights;
    double bias=0;
    double z;
    double output;
    double delta; //for backpropagation appaerently
};

struct Layer {
    vector<Node> nodes;
};


vector<AccountFeatures> parseCSV(const string& filename){
    // Implementation of CSV parsing
    vector<AccountFeatures> Acf;

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return Acf;
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
        string account_id = fields[0];
        int in_degree = fields[1];
        int out_degree = fields[2];
        double avg_neighbor_degree = fields[3];
        double pagerank = fields[4];
        int is_fraud = stoi(fields[5]);
        
        Acf.push_back({in_degree, out_degree, avg_neighbor_degree, is_fraud, pagerank, 0.0});
    }

    return Acf;
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

struct NeuralNetwork{
    vector<Layer> network[5];
    
    // Initialize network with random weights
    vector<double> w1 = initialize_weights(5);
    vector<double> w2 = initialize_weights(16);
    vector<double> w3 = initialize_weights(8);
    vector<double> w4 = initialize_weights(4);
    vector<double> w5 = initialize_weights(2);

    network[0].nodes.resize(16);
    network[1].nodes.resize(8);
    network[2].nodes.resize(4);
    network[3].nodes.resize(2);
    network[4].nodes.resize(1);

    network[0].nodes[0].weights = w1;
    network[1].nodes[0].weights = w2;
    network[2].nodes[0].weights = w3;
    network[3].nodes[0].weights = w4;
    network[4].nodes[0].weights = w5;
}