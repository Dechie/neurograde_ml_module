#include<bits/stdc++.h>
#define rep(i,n) for(int i=0;i<(int)(n);i++)
#define rrep(i,n) for(int i=(int)(n)-1;i>=0;i--)
#define each(a,x) for(auto a : (x))
#define all(a) (a).begin(),(a).end()
#define chmin(a,b) ((a) = min((a),(b)))
#define chmax(a,b) ((a) = max((a),(b)))
#define in_range(x,l,r) ((l)<=(x) && (x)<(r))
#define printvec(a) rep(i,a) cout << a[i] << " \n"[i+1==(a).size()];
#define fs first
#define sc second
#define em emplace
#define eb emplace_back
#define sz size()
#define MP make_pair
using namespace std;
typedef long long ll;
typedef double D;
typedef pair<int,int> pii;
typedef pair<ll,ll> pll;
typedef vector<int> vi;
typedef vector<ll> vl;
typedef vector<string> vs;

const ll INF = 1e8;
const D EPS = 1e-8;
const ll MOD = 998244353;

int main(){
  ll a, b, c, d;
  cin >> a >> b >> c >> d;
  vector< vector< vector< vector<ll> > > > dp(3, vector< vector< vector<ll> > >(d+1, vector< vector<ll> >(3, vector<ll>(3, 0))));
  dp[a%3][b][0][0] = 1;

  for (int i = a; i <= c; ++i) {
    int cur = i % 3;
    int prv = (cur + 2) % 3;
    for (int j = b; j <= d; ++j) {
      dp[cur][j][0][1] = (i-1) * (dp[cur][j-1][0][0] + dp[cur][j-1][0][1] + dp[cur][j-1][0][2]);
      dp[cur][j][0][1] %= MOD;
      dp[cur][j][1][0] = (j-1) * (dp[prv][j][0][0] + dp[prv][j][1][0] + dp[prv][j][2][0]);
      dp[cur][j][1][0] %= MOD;

      dp[cur][j][1][1] = max((j-1) * (dp[prv][j][0][1] + dp[prv][j][1][1] + dp[prv][j][2][1]), (i-1) * (dp[cur][j-1][1][0] + dp[cur][j-1][1][1] + dp[cur][j-1][1][2]))
	+ (dp[prv][j][0][0] + dp[prv][j][1][0] + dp[prv][j][2][0]) +  (dp[cur][j-1][0][0] + dp[cur][j-1][0][1] + dp[cur][j-1][0][2]);
      dp[cur][j][1][1] %= MOD;
      dp[cur][j][1][2] = dp[prv][j][0][1] + dp[prv][j][1][1] + j * dp[prv][j][1][2] + dp[prv][j][2][1] + j * dp[prv][j][2][2];
      dp[cur][j][1][2] %= MOD;
      dp[cur][j][2][1] = dp[cur][j-1][1][0] + dp[cur][j-1][1][1] + dp[cur][j-1][1][2] + i * dp[cur][j-1][2][1] + i * dp[cur][j-1][2][2];
      dp[cur][j][2][1] %= MOD;
    }
    for (int j = b; j <= d; ++j) {
      rep(x,3) {
	rep(y, 3){
	  dp[prv][j][x][y] = 0;
	}
      }
    }
  }   

  ll ans = 0;
  rep(i, 3) rep(j, 3) ans += dp[c%3][d][i][j];
  cout << ans % MOD << endl;
  return 0;
}
