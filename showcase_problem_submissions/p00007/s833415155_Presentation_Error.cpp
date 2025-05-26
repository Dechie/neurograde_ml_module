#include<iostream>
using namespace std;

int main(){
int n,M,m;
M=100;
cin>>n;
for(int i=0;i<n;i++){
m=(M+19)/20;
M+=m;
}
cout<<M*1000;
}