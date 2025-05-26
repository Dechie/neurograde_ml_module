#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll INF = 1e18;
const int inf = 1e9;
double pi = 3.14159265359;
#define rep(i, a, b) for (int i = a; i < b; i++)
#define per(i, b, a) for (int i = a - 1; i >= b; i--)
using Graph = vector<vector<int>>;
using pint = pair<int, int>;
int dx[4] = {1, 0, -1, 0}, dy[4] = {0, 1, 0, -1};
int dxx[8] = {1, 1, 1, 0, 0, -1, -1, -1}, dyy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

int a[205];
bool rec (int i, int n, int m) {
    if (m == 0) return true;
    if (i >= n) return false;
    bool res = rec (i + 1, n, m) || rec (i + 1, n, m - a[i]);
    return res;
}

int main() {
    int n;
    cin >> n;
    rep (i, 0, n) cin >> a[i];
    int q;
    cin >> q;
    rep (i, 0, q) {
        int m;
        cin >> m;
        if (rec (0, n, m)) cout << "yes\n";
        else
            cout << "no\n";
    }
}
