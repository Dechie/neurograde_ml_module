#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using vi = vector<int>;
using vvi = vector<vi>;
using vll = vector<ll>;
using vqi = vector<queue<int>>;

#define rep(i, n) for (int i = 0; i < (int)(n); i++)

void dfs(vqi &g, vi &d, vi &f, int v, int &t) {
  if (d[v] && g[v].empty()) {
    t--;
    return;
  }

  if (!d[v]) d[v] = t;

  while (!g[v].empty()) {
    int next = g[v].front(); g[v].pop();
    if (!d[next] )dfs(g, d, f, next, ++t);
  }

  f[v] = max(++t, f[v]);
}

int main() {
  int n;
  cin >> n;
  
  vqi g(n);
  rep(i, n) {
    int u, k;
    cin >> u >> k;

    rep(j, k) {
      int v;
      cin >> v;
      g[u - 1].push(v - 1);
    }
  }

  vi d(n), f(n);
  stack<int> s;
  s.push(0);

  int t = 1;
  rep(i, n) {
    dfs(g, d, f, i, t);
    t++;
  }

  rep(i, n) printf("%d %d %d\n", i + 1, d[i], f[i]);
}

