#pragma GCC optimize ("O3")
#include <bits/stdc++.h>
#define DEBUG true
#ifdef ONLINE_JUDGE
#undef DEBUG
#define DEBUG false
#endif 

using namespace std;

#define MAXN ((int)2e3+5)
#define MOD ((int)1e9 + 7)
#define INF ((int)1e9 + 9)
#define ll long long
#define _ << " " <<
#define CLEAR(a, b) memset(a, b, sizeof(a))
#define TRACE(x) if(DEBUG) cerr << #x << " = " << x << endl;
#define TRACE2(x,y) if(DEBUG) cerr << #x << " = " << x << " | " << #y << " = " << y << endl;
#define pb push_back
#define all(x) x.begin(), x.end()
#define endl "\n"
#define pii pair<int,int>
#define mid ((l+r)/2)

int n,m;
int a[MAXN];
int b[MAXN];

int dp[MAXN][MAXN];

map <int,int> mpa[100*MAXN];
map <int,int> mpb[100*MAXN];

ll rec(int x,int y)
{
	if(x == 0 and y == 0)
		return (a[0] == b[0] ? 1 : 0);
	else if(x == 0 or y == 0)
	{
		if(x==0)
			return mpb[y][a[0]]%MOD;
		return mpa[x][b[0]]%MOD;
	}
	if(dp[x][y] != -1) return dp[x][y];

	ll p1 = ((a[x] == b[y]) ? rec(x-1,y-1) + 1: 0);
	ll p2 = rec(x-1,y);
	ll p3 = rec(x,y-1);
	ll p4 = rec(x-1,y-1);


	return dp[x][y] = (((p1 + p2)%MOD + p3)%MOD - p4 + MOD) % MOD;
}


int main()
{
	ios_base::sync_with_stdio(false);cin.tie(0);
	cin >> n >> m;


	
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			dp[i][j] = -1;

	for (int i = 0; i < n; ++i)
	{
		cin >> a[i];
		if(i)	
		{
			mpa[i] = mpa[i-1];
			mpa[i][a[i]]++;
		}
		else
			mpa[i][a[i]]++;

	}
	for (int i = 0; i < m; ++i)
	{
		cin >> b[i];
		if(i)	
		{
			mpb[i] = mpb[i-1];
			mpb[i][b[i]]++;
		}
		else
			mpb[i][b[i]]++;
	}

	rec(n-1,m-1);

	cout << (dp[n-1][m-1] + 1) % MOD << endl;

}