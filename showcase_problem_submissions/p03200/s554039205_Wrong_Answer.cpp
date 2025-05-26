#include <iostream> 
#include <string> 
using namespace std;
int main(){
    int x=0, score=0;
    string a;
    cin >> a;
    for(int i=0; i<a.size(); i++){
        if(a[i]=='B'){
            x++;
        }
        else{
            score+=x;
        }
    }
    cout << score << endl;
    return 0;
}