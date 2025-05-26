#include<iostream>
using namespace std;
int main(){
  int n,ans = 100000;
  cin >>n;
  while(n--){
    ans = (int)(ans*1.05);
    if(ans%1000){ans = (ans/1000+1)*1000;}
  }
  cout <<ans;
  return 0;
}