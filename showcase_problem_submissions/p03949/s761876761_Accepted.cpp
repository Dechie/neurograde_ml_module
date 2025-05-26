#include <bits/stdc++.h>
using namespace std;
#define int long long
const int MOD = 1000000007;
int mx[100000],mi[100000];
bool k[100000]={};
bool used[100000]={};
vector<int> hen[100000];
void dfsx(int i){
  used[i]=true;
  for(int j=0;j<hen[i].size();j++){
    if(used[hen[i][j]])continue;
    
   if(!k[hen[i][j]]) mx[hen[i][j]]=min(mx[hen[i][j]],mx[i]+1);
    dfsx(hen[i][j]);
    if(!k[i])mx[i]=min(mx[hen[i][j]]+1,mx[i]);
  }
}
void dfsi(int i){
  used[i]=true;
  for(int j=0;j<hen[i].size();j++){
    if(used[hen[i][j]])continue;
     if(!k[hen[i][j]])  mi[hen[i][j]]=max(mi[hen[i][j]],mi[i]-1);
    dfsi(hen[i][j]);
   if(!k[i]) mi[i]=max(mi[hen[i][j]]-1,mi[i]);
  }
}
bool dfs(int i){
  if(mi[i]>mx[i])return false;
  used[i]=true;
  for(int j=0;j<hen[i].size();j++){
    if(used[hen[i][j]])continue;
    if(abs(mx[hen[i][j]]-mx[i])!=1)return false;
    if(!dfs(hen[i][j]))return false;
  }
  return true;
}
signed main() {
 int n;
 cin>>n;
 fill(mx,mx+n,MOD);fill(mi,mi+n,-1*MOD);
 int a,b;
 for(int i=0;i<n-1;i++){
   cin>>a>>b;
   
   hen[a-1].push_back(b-1);
   hen[b-1].push_back(a-1);

 }
 

 int K,v,p;
 cin>>K;
 for(int i=0;i<K;i++){
   cin>>v>>p;
   k[v-1]=true;
   mx[v-1]=p;
   mi[v-1]=p;
 }
 fill(used,used+n,false);
 dfsx(0);
fill(used,used+n,false);
  dfsx(0);
  fill(used,used+n,false);
 dfsi(0);
fill(used,used+n,false);
  dfsi(0);
  fill(used,used+n,false);
if(dfs(0)){
  cout<<"Yes"<<endl;
  for(int i=0;i<n;i++)cout<<mx[i]<<endl;
}else{
  cout<<"No";
 // for(int i=0;i<n;i++)cerr<<mx[i]<<endl;
}
return 0;
}
