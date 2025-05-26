#include <bits/stdc++.h>
using namespace std;

#define debug(x) cout << #x << ": " << (x) << endl

typedef vector<pair<int, vector<int> > > Graph;

const int INF = INT_MAX;

bool dfs(Graph &G, int par, int v, int upper, int lower,
         pair<int, int> range[]) {
  int P = G[v].first;

  if (P != INF) {
    if (((upper & 1) == (P & 1)) and lower <= P and P <= upper)
      upper = lower = P;
    else
      return false;
  }

  for (auto u: G[v].second) {
    if (u == par) continue;
    if (not dfs(G, v, u, upper + 1, lower - 1, range))
      return false;
    upper = min(upper, range[u].first + 1);
    lower = max(lower, range[u].second - 1);
  }

  range[v] = make_pair(upper, lower);
  return true;
}

void mark(Graph *G, int root, pair<int, int> range[]) {
  stack<pair<int, int> > s;
  s.push(make_pair(root, root));

  while (not s.empty()) {
    auto pv = s.top(); s.pop();

    int par = pv.first;
    int v = pv.second;
    int P = G->at(par).first;
    if (G->at(v).first == INF)
      G->at(v).first = P + 1 <= range[v].first ? P + 1 : P - 1;

    for (auto u: G->at(v).second) {
      if (u == par) continue;
      s.push(make_pair(v, u));
    }
  }
}

int main(int argc, char *argv[]) {
  int N, K;
  cin >> N;

  int A[N], B[N];
  for (int i = 0; i < N - 1; ++i) {
    cin >> A[i] >> B[i];
    A[i]--;
    B[i]--;
  }

  cin >> K;
  int V[K], P[K];
  for (int i = 0; i < K; ++i) {
    cin >> V[i] >> P[i];
    V[i]--;
  }

  Graph G(N);
  for (int i = 0; i < N - 1; ++i) {
    G[A[i]].second.push_back(B[i]);
    G[B[i]].second.push_back(A[i]);
    G[A[i]].first = INF;
    G[B[i]].first = INF;
  }

  for (int i = 0; i < K; ++i) {
    G[V[i]].first = P[i];
  }

  pair<int, int> range[N];
  bool is_possible = dfs(G, -1, V[0], P[0], P[0], range);

  if (is_possible) {
    mark(&G, V[0], range);
    cout << "Yes" << endl;
    for (int i = 0; i < N; ++i) {
      cout << G[i].first << endl;
    }
  } else {
    cout << "No" << endl;
  }

  return 0;
}
