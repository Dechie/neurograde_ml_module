#include <bits/stdc++.h>
using namespace std;

#define mp make_pair
#define pb push_back

typedef long long ll;
typedef pair<int, int> ii;
typedef pair<ll, ll> l4;

const int maxn = 2e3+1;

const int mod = 1e9+7;
ll d[maxn][maxn];
inline void Add(ll &x, ll y)
{
  if ((x += y) >= mod) x -= mod;
}
inline void Sub(ll &x, ll y)
{
  if ((x -= y) < 0) x += mod;
}

vector<int> read(int x)
{
  vector<int> ret(x);
  for (auto & e : ret) cin >> e;
  return ret;
}
int main()
{
  int n, m; cin >> n >> m;
  auto s = read(n);
  auto t = read(m);
  for (int i = 0; i < maxn; ++i)
    d[i][0] = d[0][i] = 1;
  for (int i = 1; i <= s.size(); ++i)
    for (int j = 1; j <= t.size(); ++j)
      {
	Add(d[i][j] = d[i-1][j], d[i][j-1]);
	if (s[i-1] != t[j-1]) Sub(d[i][j], d[i-1][j-1]);
	//	cerr << i << " " << j << " " << d[i][j] << endl;
      }
  printf("%lld\n", d[s.size()][t.size()]);
}
