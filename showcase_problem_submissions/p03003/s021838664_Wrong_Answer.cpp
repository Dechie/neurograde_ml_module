#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
using namespace std;

int main(void){
    long long MOD = 1000000007;
    int n, m;
    cin >> n >> m;
    vector<int> s(n, 0);
    for(int i = 0; i < n; i++){
        cin >> s[i];
    }
    vector<int> t(m, 0);
    for(int i = 0; i < m; i++){
        cin >> t[i];
    }

    vector<vector<long long>> dp(n + 1, vector<long long>(m + 1, 0));
    vector<vector<long long>> sumdp(n + 1, vector<long long>(m + 1, 1));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            if(s[i] == t[j]){
                dp[i + 1][j + 1] = sumdp[i][j] % MOD;
            }
            sumdp[i + 1][j + 1] = (sumdp[i + 1][j] + sumdp[i][j + 1] - sumdp[i][j] + dp[i + 1][j + 1]) % MOD;
        }
    }
    long long ans = 1;
    for(int i = 0; i <= n; i++){
        for(int j = 0; j <= m; j++){
            ans += dp[i][j] % MOD;
            ans %= MOD;
        }
    }
    cout << ans << endl;
}
