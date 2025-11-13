//this file performs necessary steps to get the desired outputs which will be sent to the user interface for visualising 

#include<iostream>
#include<queue>
#include<vector>
#include<unordered_map>
#include<fstream> //for file handling
#include<sstream> //to break a string into different variables
#include<algorithm>
using namespace std;

struct transaction{
    string id;
    float risk;
};

vector<transaction> transacts;
unordered_map<string, float> riskmap;
priority_queue<pair<float, string>> topriskyaccs;

//loaddata reads the data from the csv file and stores it in vector transacts
void loaddata(){
    ifstream file("results/risk_scores.csv");
    if(!file.is_open()){
        cout << "Error: could not open data/risk_scores.csv" << endl;
        return;
    }

    string line;
    getline(file, line); // Skip header (user_id,risk_score)
    
    int count = 0;
    while(getline(file, line)){ 
        // Skip empty lines
        if(line.empty()) continue;
        
        stringstream ss(line);
        string id;
        float risk;
        
        getline(ss, id, ',');
        ss >> risk;
        
        // Validate data
        if(id.empty() || ss.fail()) {
            continue;
        }


        //storing all the data into different structures
        transaction x = {id, risk};
        transacts.push_back(x);
        riskmap[id] = risk;
        topriskyaccs.push({risk, id});
    }

    file.close();

    
}
void displaymenu(){
   
    cout<<endl<< "===== MAGNUS FRAUD DETECTOR DASHBOARD ====="<<endl;
    cout<< "1. Search Transaction by ID"<<endl;
    cout<< "2. Sort Transactions by Risk"<<endl;
    cout<< "3. Show Top Risky Transactions"<<endl;
    cout<< "4.  Export Dashboard data"<<endl;
    cout<< "5. Exit"<<endl;
    cout << "==========================================="<<endl;

}

void searchTransaction(string id){

    if(riskmap.count(id)){
        cout<<"Transaction found! "<<endl;
        cout<<"Account id : "<<id<<endl<<"Risk Score: "<<riskmap[id]<<endl;
    }
    else{
        cout<<"Transaction not found!"<<endl;
    }
}

vector<transaction> sortbyrisk(){ //sorting the vector by risks
    struct comparator{
        bool operator()(transaction &a, transaction &b){
            return a.risk<b.risk;
        }
    };

    sort(transacts.begin(), transacts.end(), comparator());

    return transacts;
}

void showtoprisks(){ //using max heap
    cout<<"Top 10 Risky Accounts:"<<endl;
   

    priority_queue<pair<float, string>> temp = topriskyaccs;
    for(int i = 0; i<5 && !temp.empty(); i++){
        cout<<i+1<<" - "<<"Account id: "<<temp.top().second<<"Risk score: "<<temp.top().first<<endl;
        temp.pop();
    }
    cout.flush();

}

void exportDashboardData() {
    vector<transaction> sortedTransacts = sortbyrisk();


    ofstream out("ui/dashboard.csv");
    if (!out.is_open()) {
        cout << "Error: could not open ui/dashboard.csv for writing" << endl;
        return;
    }

    // Write header
    out << "id,risk" << endl;

    // Write only top 5% transactions
    for (int i = 0; i < sortedTransacts.size(); i++) {
        out << sortedTransacts[i].id << "," << sortedTransacts[i].risk << endl;
    }

    out.close();
    cout << "Exported the total transactions to ui/dashboard.csv" << endl;
}



int main(int argc, char* argv[]){
    loaddata();
    if(argc >= 2){
        string action = argv[1];
        if(action=="search" && argc==3) searchTransaction(argv[2]);
        else if(action=="export") exportDashboardData();
        else if(action=="top") showtoprisks();
        else if(action=="sort") sortbyrisk();
        else cout << "Invalid action!" << endl;
    } else {
        cout << "No action provided. Use search/export/top" << endl;
    }
    return 0;
}

