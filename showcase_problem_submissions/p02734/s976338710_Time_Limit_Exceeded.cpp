#include<iostream>
#include<vector>
#include<cstring>
using namespace std;

const int mod = 998244353;

int main(){
    int n, s;
    cin >> n >> s;
    vector<int> v(n);
    for(int i = 0; i < n; i++)  cin >> v[i];
    int dp[3001];
    int ans = 0;
    for(int i = 0; i < n; i++){
        memset(dp, 0, sizeof(dp));
        dp[0] = 1;
        for(int j = i; j < n; j++){
            for(int k = s; k >= v[j]; k--)  (dp[k] += dp[k-v[j]]) %= mod;
            (ans += dp[s]) %= mod;
        }
    }
    cout << ans << endl;
    return 0;
}