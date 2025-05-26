#include <bits/stdc++.h>
using ll = long long;
using namespace std;
const ll MOD = 1e9+7;

int main(){
		int N,M;
		cin >> N >> M;
		vector<ll> s(N),t(M);
		for(int i=0;i<N;i++)
				cin >> s[i];
		for(int i=0;i<M;i++)
				cin >> t[i];

		vector<vector<ll> > dp(N+1,vector<ll>(M+1, 0));
		vector<vector<ll> > sum(N+1,vector<ll>(M+1, 0));

		for(int i=1;i<=N;i++){
				for(int j=1;j<=M;j++){
						if(s[i-1]==t[j-1]){
								dp[i][j] = (sum[i-1][j-1]+1) % MOD;
						}else{
								dp[i][j] = 0;
						}
						sum[i][j] = sum[i][j-1] + sum[i-1][j] - sum[i-1][j-1] + dp[i][j];
						sum[i][j] %=MOD;
				}
		}
		cout << sum[N][M] + 1 << endl;
}
