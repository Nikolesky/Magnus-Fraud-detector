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
void loaddata(){
    cout<<"soon to be loading data..";
}
void displaymenu();

void searchTransaction(){
    cout<<"searching";
}

void sorbyrisk(){
    cout<<"sorting";
}
void showtoprisks(){
    cout<<"showing";
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

void displaymenu(){
    cout << "\n===== MAGNUS FRAUD DETECTOR DASHBOARD =====\n";
    cout << "1. Search Transaction by ID\n";
    cout << "2. Sort Transactions by Risk\n";
    cout << "3. Show Top Risky Transactions\n";
    cout << "4. Exit\n";
    cout << "===========================================\n";
}