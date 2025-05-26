#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <algorithm>

using namespace std;

int main() {
    int n;
    cin >> n;
    for (int i = 0; i < n; ++i) {
        string s, t;
        cin >> s >> t;
        int dp[1001][1001] = {};
        for (int j = 0; j < s.length(); ++j) {
            for (int k = 0; k < t.length(); ++k) {
                if (s[j] == t[k]) {
                    dp[j + 1][k + 1] = max(dp[j + 1][k + 1], dp[j][k] + 1);
                }
                dp[j+1][k+1] = max(dp[j+1][k+1], dp[j+1][k]);
                dp[j+1][k+1] = max(dp[j+1][k+1], dp[j][k+1]);
            }
        }
        cout << dp[s.length()][t.length()] << endl;

    }
    return 0;
}
