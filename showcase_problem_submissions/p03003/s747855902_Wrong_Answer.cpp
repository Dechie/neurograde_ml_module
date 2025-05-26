#include <bits/stdc++.h>

using ll = long long;
using ld = long double;
constexpr ll inf = static_cast<ll>(1e17);
constexpr ll mod = static_cast<ll>(1e9 + 7);

int n, m, s[2005], t[2005];
ll dp[2005][2005];
int main() {
	std::cin >> n >> m;

	for (int i = 1; i <= n; ++i)
		std::cin >> s[i];
	for (int i = 1; i <= m; ++i)
		std::cin >> t[i];

	for (int i = 0; i <= n; ++i)
		dp[i][0] = 1;
	for (int j = 0; j <= m; ++j)
		dp[0][j] = 1;

	for (int i = 1; i <= n; ++i) {
		dp[i][0] = dp[i - 1][0];
		for (int j = 1; j <= m; ++j) {
			if (s[i] == t[j])
				dp[i][j] = (dp[i][j - 1] + dp[i - 1][j]) % mod;
			else
				dp[i][j] = (dp[i][j - 1] + dp[i - 1][j] - dp[i - 1][j - 1]) % mod;
		}
	}

	std::cout << dp[n][m] << std::endl;
	return 0;
}
