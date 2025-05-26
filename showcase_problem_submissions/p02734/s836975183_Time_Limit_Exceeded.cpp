#include <bits/stdc++.h>
using namespace std;
const int MOD=998244353;
int main(){
  int N,S;
  cin>>N>>S;
  vector<int> A(N);
  for(int i=0;i<N;i++){
    cin>>A.at(i);
  }
  int ans=0;
  int dp[3002][3002];
  for(int i=1;i<=N;i++){
    dp[i-1][0]=1;
    for(int j=i;j<=N;j++){
      for(int k=0;k<=S;k++){
        dp[j][k]=dp[j-1][k];
        if(k-A.at(j-1)>=0 && dp[j-1][k-A.at(j-1)]!=0){dp[j][k]+=dp[j-1][k-A.at(j-1)];}
      }
      ans+=dp[j][S];
      ans%=MOD;
    }
    for(int j=1;j<=N;j++){
      for(int k=0;k<=S;k++){
        dp[j][k]=0;
      }
    }
  }
  cout<<ans%MOD<<endl;
}