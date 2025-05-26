// 
// じょえチャンネル
// 高評価・チャンネル登録よろしくお願いします！
// https://www.youtube.com/watch?v=gPFr7q7eLL8
//
/*------------------------------------
　　　　　　　　　　　 ∧＿∧
　　　　　 ∧＿∧ 　（´<_｀ ）　 
　　　　 （ ´_ゝ`）　/　 ⌒i     
　　　　／　　　＼　 　  |　|
　　　 /　　 /￣￣￣￣/　　|
　 ＿_(__ﾆつ/     ＿/ .| .|＿＿＿＿
　 　　　＼/＿＿＿＿/　（u　⊃
-------------------------------------*/
#include <stdio.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <math.h>
#include <vector>
#include <stack>
#include <queue>
#include <deque>     
#include<map>
#include <set>
using namespace std;
#define int long long
#define ll long long
#define ld long double
#define rep(i,n) for(ll i=0;i<n;++i)
#define Rep(i,n) for(ll i=1;i<=n;++i)
#define loop(i,m,n) for(ll i=m;i<n;++i)
#define rev(i,n) for(ll i=n-1;i>=0;--i)
#define ggr getchar(); getchar();return 0;
#define pie 3.141592653589793238462643383279
#define elif else if
#define P pair<ll,ll>
#define prique priority_queue<ll>
#define all(a)  (a).begin(),(a).end()
const int mod = 1000000007;
const int mod2 = 998244353;
const int inf = 1 << 30;
const int MOD = 1e9 + 7;
long long fac[510000], finv[510000], inv[510000];
long long MAX = 510000;
bool COMinited = false;
long long COM(int n, int k) {
	if (COMinited == false) {
		fac[0] = fac[1] = 1;
		finv[0] = finv[1] = 1;
		inv[1] = 1;
		for (int i = 2; i < MAX; i++) {
			fac[i] = fac[i - 1] * i % MOD;
			inv[i] = MOD - inv[MOD % i] * (MOD / i) % MOD;
			finv[i] = finv[i - 1] * inv[i] % MOD;
		}
		COMinited = true;
	}
	if (n < k) return 0;
	if (n < 0 || k < 0) return 0;
	return fac[n] * (finv[k] * finv[n - k] % MOD) % MOD;
}
template<int MOD> struct Fp {
	long long val;
	constexpr Fp(long long v = 0) noexcept : val(v% MOD) {
		if (val < 0) val += MOD;
	}
	constexpr int getmod() { return MOD; }
	constexpr Fp operator - () const noexcept {
		return val ? MOD - val : 0;
	}
	constexpr Fp operator + (const Fp& r) const noexcept { return Fp(*this) += r; }
	constexpr Fp operator - (const Fp& r) const noexcept { return Fp(*this) -= r; }
	constexpr Fp operator * (const Fp& r) const noexcept { return Fp(*this) *= r; }
	constexpr Fp operator / (const Fp& r) const noexcept { return Fp(*this) /= r; }
	constexpr Fp& operator += (const Fp& r) noexcept {
		val += r.val;
		if (val >= MOD) val -= MOD;
		return *this;
	}
	constexpr Fp& operator -= (const Fp & r) noexcept {
		val -= r.val;
		if (val < 0) val += MOD;
		return *this;
	}
	constexpr Fp& operator *= (const Fp & r) noexcept {
		val = val * r.val % MOD;
		return *this;
	}
	constexpr Fp& operator /= (const Fp & r) noexcept {
		long long a = r.val, b = MOD, u = 1, v = 0;
		while (b) {
			long long t = a / b;
			a -= t * b; swap(a, b);
			u -= t * v; swap(u, v);
		}
		val = val * u % MOD;
		if (val < 0) val += MOD;
		return *this;
	}
	constexpr bool operator == (const Fp & r) const noexcept {
		return this->val == r.val;
	}
	constexpr bool operator != (const Fp & r) const noexcept {
		return this->val != r.val;
	}
	friend constexpr ostream& operator << (ostream & os, const Fp<MOD> & x) noexcept {
		return os << x.val;
	}
	friend constexpr Fp<MOD> modpow(const Fp<MOD> & a, long long n) noexcept {
		if (n == 0) return 1;
		auto t = modpow(a, n / 2);
		t = t * t;
		if (n & 1) t = t * a;
		return t;
	}
};
using mint = Fp<1000000007>;
int bin(int bina) {
	int ans = 0;
	for (int i = 0; bina > 0; i++)
	{
		ans = ans + (bina % 2) * pow(10, i);
		bina = bina / 2;
	}
	return ans;
}
bool prime(int n) {
	for (ll i = 2; i <= sqrt(n); i++) {
		if (n % i == 0)return false;
	}
	return n != 1;
}
int gcd(int x, int y) {
	if (y == 0)return x;
	return gcd(y, x % y);
}
int lcm(int x, int y) {
	return x * y / gcd(x, y);
}
int kai(int x, int y) {
	int res = 1;
	for (int i = x - y + 1; i <= x; i++) {
		res *= i; res %= mod;
	}
	return res;
}
int mod_pow(int x, int y, int m) {
	int res = 1;
	while (y > 0) {
		if (y & 1) {
			res = res * x % m;
		}
		x = x * x % m;
		y >>= 1;
	}
	return res;
}
int jo(int x, int y) {
	int res = 1;
	while (y > 0) {
		if (y & 1) {
			res = res * x;
		}
		x = x * x;
		y >>= 1;
	}
	return res;
}

int per(int n, int r) {
	r = n - r;
	int sum = 1;
	int i;
	for (i = n; i >= r + 1; i--)sum *= i;
	return sum;
}
struct UnionFind {
	vector<int> par;

	UnionFind(int n) : par(n, -1) { }

	int root(int x) {
		if (par[x] < 0) return x;
		else return par[x] = root(par[x]);
	}

	bool issame(int x, int y) {
		return root(x) == root(y);
	}

	bool merge(int x, int y) {
		x = root(x); y = root(y);
		if (x == y) return false;
		if (par[x] > par[y]) swap(x, y);
		par[x] += par[y];
		par[y] = x;
		return true;
	}

	int size(int x) {
		return -par[root(x)];
	}
};
struct Eratos {
	vector<int> primes;
	vector<bool> isprime;
	vector<int> min_factor;

	Eratos(int MAX) : primes(),
		isprime(MAX + 1, true),
		min_factor(MAX + 1, -1) {
		isprime[0] = isprime[1] = false;
		min_factor[0] = 0, min_factor[1] = 1;
		for (int i = 2; i <= MAX; ++i) {
			if (!isprime[i]) continue;
			primes.push_back(i);
			min_factor[i] = i;
			for (int j = i * 2; j <= MAX; j += i) {
				isprime[j] = false;
				if (min_factor[j] == -1) min_factor[j] = i;
			}
		}
	}
	// prime factorization
	vector<pair<int, int>> prime_factorize(int n) {
		vector<pair<int, int> > res;
		while (n != 1) {
			int prime = min_factor[n];
			int exp = 0;
			while (min_factor[n] == prime) {
				++exp;
				n /= prime;
			}
			res.push_back(make_pair(prime, exp));
		}
		return res;
	}
};
template<class T>
vector<T> make_vec(size_t a) {
	return vector<T>(a);
}
template<class T, class... Ts>
auto make_vec(size_t a, Ts... ts) {
	return vector<decltype(make_vec<T>(ts...))>(a, make_vec<T>(ts...));
}
void add(long long& a, long long b) {
	a += b;
	if (a >= MOD) a -= MOD;
}
#define F first
#define S second
template<class T> inline bool chmax(T& a, T b) { if (a < b) { a = b; return 1; } return 0; }
template<class T> inline bool chmin(T& a, T b) { if (a > b) { a = b; return 1; } return 0; }
//cout << fixed << setprecision(10) << num << endl;
int a[55];
signed main() {
	int n;
	cin >> n;
	rep(i, n) {
		cin >> a[i];
	}
	int ans = 0;
	rep(k, 100010) {
		rep(i, n) {
			if (a[i] >= n) {
				int dec = a[i] / n;
				ans += dec;
				a[i] -= dec * n;
				rep(j, n) {
					if (i != j) a[j] += dec;
				}
			}
		}
	}
	cout << ans << endl;
}