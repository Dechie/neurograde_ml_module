#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <functional>
#include <queue>
#include <string>
#include <cstring>
#include <numeric>
#include <cstdlib>
#include <cmath>
#include <map>
#include <unordered_map>
#include <set>
using namespace std;

typedef long long ll;

#define INF 10e17 // 4倍しても(4回足しても)long longを溢れない
#define rep(i,n) for(int i=0; i<n; i++)
#define rep_r(i,n,m) for(int i=m; i<n; i++)
#define END cout << endl
#define MOD 1000000007
#define pb push_back
#define sorti(x) sort(x.begin(), x.end())
#define sortd(x) sort(x.begin(), x.end(), std::greater<int>())
#define debug(x) std::cerr << (x) << std::endl;
#define roll(x) for (auto itr : x) { debug(itr); }

template <class T> inline void chmax(T &ans, T t) { if (t > ans) ans = t;}
template <class T> inline void chmin(T &ans, T t) { if (t < ans) ans = t;}

ll powmod(ll n, ll k) {
  ll ret = 1;
  while (k) {
    if (k & 1) ret = ret * n % MOD;
    n = n * n % MOD;
    k >>= 1;
  }
  return ret;
}

int main() {
  string s;
  cin >> s;
  ll ans = 0;
  ll cnt = 0;
  int n = s.size();
  for (int i = 0; i < n; ++i) {
    if (s[i] == '0') continue;
    ans += powmod(2, cnt) * powmod(3, n - i - 1) % MOD;
    cnt++;
  }
  ans += powmod(2, cnt);
  cout << ans % MOD << endl;
  return 0;
}
