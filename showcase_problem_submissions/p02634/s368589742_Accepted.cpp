#include <bits/stdc++.h>
#define _overload3(_1,_2,_3,name,...)name
#define _rep(i,n)repi(i,0,n)
#define repi(i,a,b)for(int i=int(a),i##_len=(b);i<i##_len;++i)
#define MSVC_UNKO(x)x
#define rep(...)MSVC_UNKO(_overload3(__VA_ARGS__,repi,_rep,_rep)(__VA_ARGS__))
#define all(c)c.begin(),c.end()
#define write(x)cout<<(x)<<'\n'
using namespace std; using ll = long long; template<class T>using vv = vector<vector<T>>;
template<class T>auto vvec(int n, int m, T v) { return vv<T>(n, vector<T>(m, v)); }
constexpr int INF = 1 << 29, MOD = 998244353; constexpr ll LINF = 1LL << 60;
struct aaa { aaa() { cin.tie(0); ios::sync_with_stdio(0); cout << fixed << setprecision(10); }; }aaaa;


template<std::uint_fast64_t Mod> class modint {
    using uint = std::uint_fast64_t;
public:
    uint a;
    modint() noexcept : a(0) {}
    constexpr modint(const uint x) noexcept : a(x% Mod) {}
    constexpr uint& value() noexcept { return a; }
    constexpr const uint& value() const noexcept { return a; }
    constexpr modint operator +(modint rhs) const noexcept { return modint(*this) += rhs; }
    constexpr modint operator -(modint rhs) const noexcept { return modint(*this) -= rhs; }
    constexpr modint operator *(modint rhs) const noexcept { return modint(*this) *= rhs; }
    constexpr modint operator /(modint rhs) const noexcept { return modint(*this) /= rhs; }
    constexpr modint& operator +=(modint rhs) noexcept { a += rhs.a; if (a >= Mod) a -= Mod; return *this; }
    constexpr modint& operator -=(modint rhs) noexcept { if (a < rhs.a) a += Mod; a -= rhs.a; return *this; }
    constexpr modint& operator *=(modint rhs) noexcept { a = a * rhs.a % Mod; return *this; }
    constexpr modint& operator /=(modint rhs) noexcept {
        for (uint exp = Mod - 2; exp; exp /= 2, rhs *= rhs) if (exp % 2) *this *= rhs;
        return *this;
    }
};
template<std::uint_fast64_t Mod, typename T> constexpr modint<Mod> pow(modint<Mod> base, T exp) noexcept {
    modint<Mod> y(1); for (; exp; exp /= 2, base *= base) if (exp % 2) y *= base;
    return y;
}
template<std::uint_fast64_t Mod> std::istream& operator >>(std::istream& lhs, modint<Mod>& rhs) { return lhs >> rhs.a; }
template<std::uint_fast64_t Mod> std::ostream& operator <<(std::ostream& lhs, modint<Mod> rhs) { return lhs << rhs.a; }
using mint = modint<MOD>;

int main() {
    int A, B, C, D;
    cin >> A >> B >> C >> D;

    int E = C - A, F = D - B;
    vv<mint> dp = vvec(E + 1, F + 1, mint(0));
    dp[0][0] = 1;

    rep(i, E) {
        mint s = 0;
        rep(j, F + 1) {
            dp[i + 1][j] += s + dp[i][j] * (B + j);
            s += dp[i][j];
            s *= A + i;
        }
    }

    mint ans = 0;
    rep(j, F + 1) {
        ans += dp[E][j] * pow(mint(C), F - j);
    }
    write(ans);
}
