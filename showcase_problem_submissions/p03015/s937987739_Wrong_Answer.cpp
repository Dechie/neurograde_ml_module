#include<bits/stdc++.h>
#include<iomanip>

using namespace std;
using ll = long long;
constexpr ll mod = 1e9+7;
constexpr ll md = mod;
constexpr ll inf = 1e15;
int main(){
    string L;
    cin>>L;
    int N = (int)L.size();
    vector<vector<int>> dp(N+5,vector<int>(2));
    dp[0][0] = 1;
    for(int i=0;i<N;++i){
        if(L[i] == '1'){
            dp[i+1][0] = (dp[i][0] * 2)%mod; 
            dp[i+1][1] = ((dp[i][1] * 3)%mod + dp[i][0])%mod;
        }else{
            dp[i+1][0] = dp[i][0];
            (dp[i+1][1] += dp[i][1] * 3 %mod) %= mod;
        }
    }
    cout << dp[N][0] + dp[N][1] << endl;
    return 0;
}

