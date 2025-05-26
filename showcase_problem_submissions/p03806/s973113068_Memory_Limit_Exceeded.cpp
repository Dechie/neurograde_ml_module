#include<bits/stdc++.h>
typedef int ll;
typedef unsigned long long ull;
using namespace std;
#define pb push_back
int dy[]={0, 0, 1, -1, 1, 1, -1, -1};
int dx[]={1, -1, 0, 0, 1, -1, -1, 1};

#define FOR(i,a,b) for (int i=(a);i<(b);i++)
#define RFOR(i,a,b) for (int i=(b)-1;i>=(a);i--)
#define REP(i,n) for (int i=0;i<(n);i++)
#define RREP(i,n) for (int i=(n)-1;i>=0;i--)
#define mp make_pair
#define fi first
#define sc second
#define INF 1000000000
ll gcd (ll a,ll b) {
	if(b == 0) {
		return a;
	}
	return gcd(b,a % b);
}

ll n,ma,mb,a[100],b[100],c[100];
ll dp[41][2500][2500];

int main(){

	cin >> n;

	cin >> ma >> mb;

	REP(i,n) {
		cin >> a[i] >> b[i] >> c[i];
	}

	REP(k,n + 1){
		REP(i,1001) {
			REP(j,1001) {
				dp[k][i][j] = INF;
			}
		}
	}
	dp[0][0][0] = 0;

	REP(i,n) {
		REP(k,1001) {
			REP(l,1001) {
				dp[i + 1][k + a[i]][l + b[i]] = min(dp[i + 1][k + a[i]][l + b[i]],dp[i][k][l] + c[i]);
				dp[i + 1][k][l] = min(dp[i + 1][k][l],dp[i][k][l]);
			}
		}
	}
	
	ll ans = INF;
	FOR(i,1,1001) {
		FOR(j,1,1001) {
			ll g = gcd(i,j);

			if(i / g == ma && j / g == mb) {
				ans = min(ans,dp[n][i][j]);
			}
		}
	}
	if(ans < INF) {
		cout << ans << endl;
	}else {
		cout << -1 << endl;
	}
	return 0;
}
