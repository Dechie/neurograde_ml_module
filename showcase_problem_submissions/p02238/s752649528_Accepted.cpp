#include <bits/stdc++.h>
using namespace std;

using ll = long long;
#define REP(i, n) for(int (i)=0; (i)< (n); ++i)
#define REPR(i, n) for(int (i)=(n); (i)>=0; --i)
#define FOR(i, n, m) for(int (i)=(n); (i)<(m); ++i)

int n;
vector<int> d, f;
vector<vector<int>> g;
int Time = 0;
void dfs(int v){
  d[v] = ++Time;
  REP(i, g[v].size()){
    int nv = g[v][i];
    if(d[nv] != -1) continue;
    dfs(nv);
  }
  f[v] = ++Time;
}

int main(){
  cin >> n;
  d.resize(n,-1), f.resize(n), g.resize(n);
  REP(i, n){
    int u, k;
    cin >> u >> k;
    g[i].resize(k);
    REP(j, k){
      cin >> g[i][j];
      g[i][j]--;
    }
  }
  REP(i, n){
    if(d[i] == -1){
      dfs(i);
    }
  }
  REP(i, n){
    cout << i+1 << " " << d[i] <<" " <<  f[i] << endl;
  }
  return 0;
}

