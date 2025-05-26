#include<bits/stdc++.h>
using namespace std;

#define mp make_pair
#define pb push_back
#define x first
#define y second
#define all(a) (a).begin(), (a).end()
typedef long long ll;

const int mod = 998244353;

int add(int a, int b) {
    return a + b < mod ? a + b : a + b - mod;
}

int sub(int a, int b) {
    return a >= b ? a - b : a - b + mod;
}

int mult(int a, int b) {
    return (ll)a * b % mod;
}

int dp[3005][3005];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    int a, b, c, d;
    cin >> a >> b >> c >> d;
    dp[a][b] = 1;
    for (int i = a; i <= c; i++) {
        for (int j = b; j <= d; j++) {
            if (i + 1 <= c) dp[i + 1][j] = add(dp[i + 1][j], mult(dp[i][j], j));
            if (j + 1 <= d) dp[i][j + 1] = add(dp[i][j + 1], mult(dp[i][j], i));
            if (i + 1 <= c && j + 1 <= d) dp[i + 1][j + 1] = sub(dp[i + 1][j + 1], mult(dp[i][j], mult(i, j)));
        }
    }
    cout << dp[c][d];
    return 0;
}
