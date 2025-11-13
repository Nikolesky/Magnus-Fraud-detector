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

//the functions i want in my dashboard


//loaddata reads the data from the csv file and stores it in vector transacts
void loaddata(){
    ifstream file("data/risk_scores.csv");
    if(!file.is_open()){
        cout<<"Error: could not open risk_scores.csv \n";
        return;
    }

    string line;
    getline(file, line);

    while(getline(file, line)){ //breaks each line into separate variables 
        stringstream ss;
        string id;
        float risk;
        getline(ss, id, ',');
        ss>>risk;

        transaction x = {id, risk};
        transacts.push_back(x);
        riskmap[id] = risk;
        topriskyaccs.push({risk, id});
    }

    file.close();

    cout<<"Successfully loaded data... \n"<<transacts.size()<<" records found\n";
    
}
void displaymenu(){
   
    cout << "\n===== MAGNUS FRAUD DETECTOR DASHBOARD =====\n";
    cout << "1. Search Transaction by ID\n";
    cout << "2. Sort Transactions by Risk\n";
    cout << "3. Show Top Risky Transactions\n";
    cout << "4. Exit\n";
    cout << "===========================================\n";

}

void searchTransaction(){
    string id;
    cout<<"Enter the id you want to search for: ";
    cin>>id;

    if(riskmap.count(id)){
        cout<<"Transaction found! \n";
        cout<<"Account id : "<<id<<"\nRisk Score: "<<riskmap[id]<<endl;
    }
    else{
        cout<<"Transaction not found!\n";
    }
}

void sorbyrisk(){ //sorting the vector by risks
    struct comparator{
        bool operator()(transaction &a, transaction &b){
            return a.risk<b.risk;
        }
    };

    sort(transacts.begin(), transacts.end(), comparator());

    cout<<"Transactions in descending order..";
    int n = transacts.size();
    for(int i = 0; i < min(n, 10); i++){
        cout<<i+1<<"."<<" Account id: "<<transacts[i].id<<" Risk: "<<transacts[i].risk<<endl;
    }
    
}

void showtoprisks(){
    cout<<"Top 10 Risky Accounts:";
    cout<<"Loading..";

    priority_queue<pair<float, string>> temp = topriskyaccs;
    for(int i = 0; i<10 && !temp.empty(); i++){
        cout<<i+1<<" - "<<"Account id: "<<topriskyaccs.top().first<<"Risk score: "<<topriskyaccs.top().second<<endl;
        topriskyaccs.pop();
    }
    
}



int main(){
    loaddata();

    int choice;
    do{
        displaymenu();
        cout<<"Enter your choice: ";
        cin>>choice;
        cin.ignore();

        switch (choice) {
            case 1: searchTransaction(); break;
            case 2: sorbyrisk(); break;
            case 3: showtoprisks(); break;
            case 4: cout<<"Exiting Dashboars..."; break;
    
            default : cout<<"Invalid choice. Try again.";
    
        }
    }while(choice!=4);

    return 0;
}

