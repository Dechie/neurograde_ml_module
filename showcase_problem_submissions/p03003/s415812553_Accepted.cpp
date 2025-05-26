//Optimise
#include <bits/stdc++.h>
using namespace std;

// #define multitest 1
#define Debug 1
#ifdef Debug
#define db(...) ZZ(#__VA_ARGS__, __VA_ARGS__);
template <typename Arg1>
void ZZ(const char *name, Arg1 &&arg1)
{
	std::cerr << name << " = " << arg1 << endl;
}
template <typename Arg1, typename... Args>
void ZZ(const char *names, Arg1 &&arg1, Args &&... args)
{
	const char *comma = strchr(names + 1, ',');
	std::cerr.write(names, comma - names) << " = " << arg1;
	ZZ(comma, args...);
}
#else
#define db(...)
#endif

typedef long long ll;
typedef long double ld;
#define f first
#define s second
#define pb push_back
const long long mod = 1000000007;

ll power(int base, int index)
{
	if (index == 0)
		return 1;
	ll temp = power(base, index / 2);
	temp = (temp * temp) % mod;
	if (index & 1)
		temp *= base;
	return temp % mod;
}

void solve()
{
	int n, m;
	cin >> n >> m;
	vector<int> A(n), B(m);
	for (auto &x : A)
		cin >> x;
	for (auto &x : B)
		cin >> x;
	vector<vector<ll>> CommonSequences(n + 1, vector<ll>(m + 1, 1));
	ll answer = 1;
	for (int i = 1; i <= n; ++i)
		for (int j = 1; j <= m; ++j)
		{
			if (A[i - 1] == B[j - 1])
				CommonSequences[i][j] = (mod + CommonSequences[i - 1][j] + CommonSequences[i][j - 1]) % mod;
			else
				CommonSequences[i][j] = (mod + CommonSequences[i - 1][j] + CommonSequences[i][j - 1] - CommonSequences[i - 1][j - 1]) % mod;
		}
	cout << CommonSequences[n][m] % mod << '\n';
}

int main()
{
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	int t = 1;
#ifdef multitest
	cin >> t;
#endif
	while (t--)
		solve();
	return 0;
}