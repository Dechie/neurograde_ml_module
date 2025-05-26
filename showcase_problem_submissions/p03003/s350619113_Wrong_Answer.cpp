#include<bits/stdc++.h>
using namespace std ;

const int mod = 1e9 + 7 ;

const int N = 2010 ;

int s[N], t[N] ;

int n, m ; 

long long dp[N][N] ;

int main(){

	cin >> m >> n ;

	for (int i = 1; i <= m; ++ i) cin >> s[i] ;

	for (int i = 1; i <= n; ++ i) cin >> t[i] ;

	for (int i = 1; i <= m; ++ i) {
		for (int j = 1; j <= n; ++ j) {
			dp[i][j] = dp[i][j - 1] + dp[i - 1][j] - dp[i - 1][j - 1] ;
			if (s[i] == t[j]) dp[i][j] += dp[i - 1][j - 1] + 1 ;
			dp[i][j] %= mod ;
		}
	}

	cout << (dp[m][n] + 1) % mod << endl ;

	return 0 ;
}