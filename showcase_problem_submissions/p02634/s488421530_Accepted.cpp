#include<iostream>
#include<queue>
#include<vector>
int gcd(int a, int b);
using namespace std;
long long int dp[3002][3002];
long long int mod = 998244353;
long long int DP(long long int X, long long int Y);
int main() {


	//B
	long long int A, B, C, D;
	cin >> A >> B >> C >> D;
	dp[A][B] = 1;
	int i;
	for (i = A + 1; i <= C; i++) {
		dp[i][B] = (dp[i - 1][B] * B) % mod;
	}
	for (i = B + 1; i <= D; i++) {
		dp[A][i] = (dp[A][i-1] * A) % mod;
	}
	cout << DP(C, D) << endl;
	return 0;

	//A
	/*int x;
	cin >> x;
	cout << 360 / gcd(360, x) << endl;

	return 0;
}
int gcd(int a, int b) {
	if (a == 0) {
		return b;
	}
	else if (b == 0) {
		return a;
	}
	else {
		return gcd(b, a % b);
	}*/
}
long long int DP(long long int X, long long int Y) {
	//cout << X << ':' << Y << endl;
	if (dp[X][Y] > 0) {
		return dp[X][Y];
	}
	else {
		//dp[X][Y] = (((DP(X - 1, Y)*Y)%mod) + ((DP(X, Y - 1)*X)%mod)- ((DP(X-1,Y-1)*(X-1)*(Y-1))%mod)) % mod;
		dp[X][Y] = (((DP(X - 1, Y)*Y)%mod) + ((DP(X, Y - 1)*X)%mod) + mod -  ((DP(X-1,Y-1)*(X-1)*(Y-1))%mod)) % mod;

		return dp[X][Y];
	}
}
