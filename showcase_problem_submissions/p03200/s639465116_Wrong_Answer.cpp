#include<iostream>
#include<string>
using namespace std;

int main(){
  string s;
  cin >> s;
  int c=0, Ws=0;
  for(int i=0; i<s.length(); i++){
    if(s[i] == 'W'){
      c+=i-Ws;
      Ws++;
    } 
  }
    cout << c;
return 0;
}