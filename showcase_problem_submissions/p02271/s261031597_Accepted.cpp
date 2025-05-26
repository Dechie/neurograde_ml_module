#include <bits/stdc++.h>

using namespace std;

#define REP(i, n) for(int i = 0; i < n; i++)

int n;
vector<int> a(50);

int solve(int i, int m) {
    if (m == 0) return 1;
    if (i >= n || m < 0) return 0;

    int res = solve(i + 1, m) || solve(i + 1, m - a[i]);
    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    cin >> n;
    REP(i, n) cin >> a[i];
    a.resize(n);
    int q;
    cin >> q;
    REP(i, q) {
        int m;
        cin >> m;
        cout << (solve(0, m) ? "yes" : "no") << endl;
    }
    return 0;
}

