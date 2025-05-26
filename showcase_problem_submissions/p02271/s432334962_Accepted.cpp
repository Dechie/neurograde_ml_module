#include <bits/stdc++.h>
using namespace std;

// 総数を1000000007で割った余り
const long long mod = 1e9 + 7;

using ll  = long long;
using pii = pair<int, int>;
using pll = pair<ll, ll>;
#define ull unsigned long long
#define ld long double
#define vi vector<int>
#define vll vector<ll>
#define vc vector<char>
#define vs vector<string>
#define vpii vector<pii>
#define vpll vector<pll>

#define rep(i, n) for (int i = 0, i##_len = (n); i < i##_len; ++i)
#define rep1(i, n) for (int i = 1, i##_len = (n); i <= i##_len; ++i)
#define repr(i, n) for (int i = ((int)(n)-1); i >= 0; --i)
#define rep1r(i, n) for (int i = ((int)(n)); i > 0; --i)

#define sz(x) ((int)(x).size())
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()

#define SORT(v, n) sort(v, v + n);
#define VSORT(v) sort(v.begin(), v.end());
#define RSORT(x) sort(rall(x));
#define pb push_back
#define mp make_pair

#define INF (1e9)
#define PI (acos(-1))
#define EPS (1e-7)

ull gcd(ull a, ull b) { return b ? gcd(b, a % b) : a; }
ull lcm(ull a, ull b) { return a / gcd(a, b) * b; }

int n, a[50];
int solve(int i, int m) {
    // 再帰終了条件 成功条件
    if (m == 0) {
        return true;
    }
    // 再帰終了条件 失敗条件
    if (i >= n) {
        return false;
    }
    // 考えるポイント
    // 左はi番目を使わないで次へ、右はi番目（a[i]）を削って次へ。
    // つまり、i番目を使うかどうかの両方のパターンで再帰をかける。
    // ただそれだけ。
    int res = solve(i + 1, m) || solve(i + 1, m - a[i]);
    // このresを見ることで再帰でどこまで深く潜っているかわかる。
    // cout << res << " ";

    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    // cout << fixed << setprecision(5);

    int q, m, i;
    cin >> n;
    rep(i, n) cin >> a[i];
    cin >> q;
    rep(i, q) {
        cin >> m;
        // この0はiが0から始まっている。
        // mは求める数。
        if (solve(0, m))
            cout << "yes" << endl;
        else
            cout << "no" << endl;
    }

    return 0;
}

