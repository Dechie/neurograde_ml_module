#include <bits/stdc++.h>

#define rep(i, n) for (int i = 0; i < int(n); i++)
#define rrep(i, n) for (int i = int(n) - 1; i >= 0; i--)
#define reps(i, n) for (int i = 1; i <= int(n); i++)
#define rreps(i, n) for (int i = int(n); i >= 1; i--)
#define repc(i, n) for (int i = 0; i <= int(n); i++)
#define rrepc(i, n) for (int i = int(n); i >= 0; i--)
#define repi(i, a, b) for (int i = int(a); i < int(b); i++)
#define repic(i, a, b) for (int i = int(a); i <= int(b); i++)
#define repp(i, n) rep(i, n)
#define reppp(i, n) repp(i, n)
#define rrepp(i, n) rrep(i, n)
#define repss(i, n) reps(i, n)
#define repcc(i, n) repc(i, n)
#define repii(i, a, b) repi(i, a, b)
#define each(x, y) for (auto &x : y)
#define all(a) (a).begin(), (a).end()
#define bit(b) (1ll << (b))

using namespace std;

using i32 = int;
using i64 = long long;
using u64 = unsigned long long;
using f80 = long double;
using vi32 = vector<i32>;
using vi64 = vector<i64>;
using vu64 = vector<u64>;
using vf80 = vector<f80>;
using vstr = vector<string>;

inline void yes() { cout << "Yes" << '\n'; exit(0); }
inline void no() { cout << "No" << '\n'; exit(0); }
inline i64 gcd(i64 a, i64 b) { if (min(a, b) == 0) return max(a, b); if (a % b == 0) return b; return gcd(b, a % b); }
inline i64 lcm(i64 a, i64 b) { return a / gcd(a, b) * b; }
inline u64 xorshift() { static u64 x = 88172645463325252ull; x = x ^ (x << 7); return x = x ^ (x >> 9); }
void solve(); int main() { ios::sync_with_stdio(0); cin.tie(0); cout << fixed << setprecision(16); solve(); return 0; }
template <typename T> class pqasc : public priority_queue<T, vector<T>, greater<T>> {};
template <typename T> class pqdesc : public priority_queue<T, vector<T>, less<T>> {};
template <typename T> inline void amax(T &x, T y) { if (x < y) x = y; }
template <typename T> inline void amin(T &x, T y) { if (x > y) x = y; }
template <typename T> inline T power(T x, i64 n, T e = 1) { T r = e; while (n > 0) { if (n & 1) r *= x; x *= x; n >>= 1; } return r; }
template <typename T> istream& operator>>(istream &is, vector<T> &v) { each(x, v) is >> x; return is; }
template <typename T> ostream& operator<<(ostream &os, vector<T> &v) { rep(i, v.size()) { if (i) os << ' '; os << v[i]; } return os; }
template <typename T, typename S> istream& operator>>(istream &is, pair<T, S> &p) { is >> p.first >> p.second; return is; }
template <typename T, typename S> ostream& operator<<(ostream &os, pair<T, S> &p) { os << p.first << ' ' << p.second; return os; }

struct Edge {
  int to, col, dist;
  Edge(int to, int col, int dist) : to(to), col(col), dist(dist) {}
};

vector<vector<Edge>> es;

struct LowestCommonAncestor {
  int n;
  int l;
  vector<vi32> par;
  vi32 dep;
  vi64 dist;
  void dfs(int v, int p, int d, i64 di) {
    if (p != -1) par[v][0] = p;
    dep[v] = d;
    dist[v] = di;
    each(e, es[v]) {
      if (e.to == p) continue;
      dfs(e.to, v, d + 1, di + e.dist);
    }
  }
  LowestCommonAncestor(int n) : n(n) {
    l = 0;
    while ((1 << l) < n) l++;
    par = vector<vi32>(n + 1, vi32(l, n));
    dep.assign(n, 0);
    dist.assign(n, 0);
    dfs(0, -1, 0, 0);
    for (int v = 0; v < n; v++) {
      for (int i = 0; i < l; i++) {
        par[v][i + 1] = par[par[v][i]][i];
      }
    }
  }
  int lca(int u, int v) {
    if (dep[u] < dep[v]) swap(u, v);
    for (int i = 0; i < l; i++) {
      if ((1 << i) & (dep[u] - dep[v])) u = par[u][i];
    }
    if (u == v) return u;
    for (int i = l - 1; i >= 0; i--) {
      int pu = par[u][i];
      int pv = par[v][i];
      if (pu != pv) u = pu, v = pv;
    }
    return par[u][0];
  }
  i64 cost(int u, int v) {
    int c = lca(u, v);
    return dist[u] + dist[v] - dist[c] * 2;
  }
};

struct Query {
  int i, col, dist, coef;
  Query(int i, int col, int dist, int coef) : i(i), col(col), dist(dist), coef(coef) {}
};

void solve() {
  int N, Q; cin >> N >> Q;
  es = vector<vector<Edge>>(N);
  rep(i, N - 1) {
    int a, b, c, d; cin >> a >> b >> c >> d;
    a--, b--;
    es[a].emplace_back(b, c, d);
    es[b].emplace_back(a, c, d);
  }
  auto lca = LowestCommonAncestor(N);
  
  vi32 cnt(N);
  vi64 sum(N);
  vi64 ans(Q);
  vector<vector<Query>> que(N);

  rep(i, Q) {
    int x, y, u, v; cin >> x >> y >> u >> v;
    u--, v--;
    int c = lca.lca(u, v);
    que[u].emplace_back(i, x, y, 1);
    que[v].emplace_back(i, x, y, 1);
    que[c].emplace_back(i, x, y, -2);
    ans[i] = lca.cost(u, v);
  }

  function<void(int, int)> dfs = [&](int v, int p) {
    each(qu, que[v]) {
      i64 su = -sum[qu.col];
      su += (i64) qu.dist * cnt[qu.col];
      ans[qu.i] += su * qu.coef;
    }
    each(e, es[v]) {
      if (e.to == p) continue;
      cnt[e.col]++;
      sum[e.col] += e.dist;
      dfs(e.to, v);
      cnt[e.col]--;
      sum[e.col] -= e.dist;
    }
  };
  dfs(0, -1);
  rep(i, Q) {
    cout << ans[i] << '\n';
  }
}
