 #pragma GCC optimize("O3")
 #pragma GCC optimize("unroll-loops")
 #pragma GCC optimize("fast-math")
#include<bits/stdc++.h>

using namespace std;

using ll = long long;
using ld = long double;
using ull = unsigned long long;

#define int long long
#define F first
#define S second
#define all(v) v.begin(), v.end()
#define rall(v) v.rbegin(), v.rend()

void accell() {
    cin.tie(0);
    cout.tie(0);
    ios_base::sync_with_stdio(0);
}
const int N = 2001;
bitset<N * N>dp[N];

signed main() {
    accell();
    int n;
    cin >> n;
    vector<int>a(n);
    int s = 0;
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
        s += a[i];
    }
    s = (s + 1) / 2;
    dp[0][0] = 1;
    for (int i = 0; i < n; ++i) {
        dp[i + 1] |= (dp[i] << a[i]);
        dp[i + 1] |= dp[i];
    }
    int c = dp[n]._Find_next(s - 1);
    cout << c << '\n';
    return 0;
}
