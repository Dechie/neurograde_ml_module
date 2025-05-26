#include <bits/stdc++.h>

using namespace std;

struct initon {
    initon() {
        cin.tie(0);
        ios::sync_with_stdio(false);
    };
} hoee;
//別名
#define int long long
using ll = long long;
using vi = vector<int>;
using vvi = vector<vi>;
using vs = vector<string>;
using vl = vector<ll>;
using vvl = vector<vl>;
using P = pair<int, int>;
using T = tuple<int, int, int>;
using vp = vector<P>;
using dou = double;
using itn = int;
using str = string;
#define F first
#define S second
//定数
const int INF = (int) 1e9 + 100;
const int MINF = (int) -1e9 - 100;
const ll LINF = (ll) 1e18 + 100;
const ll MLINF = (ll) 1e18 - 100;
const double EPS = 1e-9;
const int y4[] = {-1, 1, 0, 0};
const int x4[] = {0, 0, -1, 1};
const int y8[] = {0, 1, 0, -1, -1, 1, 1, -1};
const int x8[] = {1, 0, -1, 0, 1, -1, 1, -1};

//配列
#define sz(a) (sizeof(a)/sizeof(a[0]))

//コンテナ
#define mp make_pair
#define pb push_back
#define eb emplace_back
#define all(a) (a).begin(),(a).end()
#define rall(a) (a).rbegin(),(a).rend()
#define sort(v) sort(v.begin(),v.end())

//繰り返し
#define _overloadrep(_1, _2, _3, name, ...) name
#define _rep(i, n) for(int i = 0,RLN = (n); i < RLN ; i++)
#define repi(i, m, n) for(int i = m,RLN = (n); i < RLN ; i++)
#define rep(...) _overloadrep(__VA_ARGS__,repi,_rep,)(__VA_ARGS__)
#define _rer(i, n) for(int RLN = (n) ,i = RLN; i >= 0 ; i--)
#define reri(i, m, n) for(int RLN = (n) ,i = m-1; i >= n ; i--)
#define rer(...) _overloadrep(__VA_ARGS__,reri,_rer,)(__VA_ARGS__)

// 多次元配列の初期化。第２引数の型のサイズごとに初期化していく。
template<typename A, size_t N, typename T>
void fill(A (&array)[N], const T &val) {
    std::fill((T *) array, (T *) (array + N), val);
}

#define arcpy(a, b) memcpy(a,b,sizeof(b))

//入力
template<typename T = int>
T in() {
    T x;
    cin >> x;
    return (x);
}

string sin() {
    return in<string>();
}

double din() {
    return in<double>();
}

ll lin() {
    return in<ll>();
}

#define na(a, n) rep(i,n) cin >> a[i];
#define nad(a, n) rep(i,n) cin >> a[i]; a[i]--;
#define nt(a, h, w) rep(hi,h)rep(wi,w) cin >> a[hi][wi];
#define ntd(a, h, w) rep(hi,h)rep(wi,w) cin >> a[hi][wi]; a[hi][wi]--;
#define nctp(a, c) fill(a,c); rep(hi,1,sz(a)+1)rep(wi,1,sz(a[0])+1) cin >> a[hi][wi];
#define add(a, n) rep(i,n) a.pb(in());


//出力
template<class T>
void out(T x) {
    if (typeid(x) == typeid(double))cout << fixed << setprecision(10) << x << endl;
    else cout << x << endl;
}
//デバッグ
#define debug(x) cerr << x << " " << "(L:" << __LINE__ << ")" << '\n';

//便利関数
#define bit(n) (1LL<<(n))

template<class T>
bool chmax(T &a, const T &b) {
    if (a < b) {
        a = b;
        return true;
    }
    return false;
}

template<class T>
bool chmin(T &a, const T &b) {
    if (b < a) {
        a = b;
        return true;
    }
    return false;
}

inline bool inside(int y, int x, int H, int W) {
    return y >= 0 && x >= 0 && y < H && x < W;
}

//mod関連

ll MOD = (int) 1e9 + 7;

class mint {
private:
    ll v;
public:
    static ll mod(ll a) { return (a % MOD + MOD) % MOD; }

    mint(ll a = 0) { this->v = mod(a); };

    mint(const mint &a) { v = a.v; }

    mint operator+(const mint &a) { return mint(v + a.v); }

    mint operator+(const ll a) { return mint(v + a % MOD); }

    mint operator+(const signed a) { return mint(v + a % MOD); }

    friend mint operator+(const ll a, const mint &b) { return mint(a % MOD + b.v); }

    void operator+=(const mint &a) { v = (v + a.v) % MOD; }

    void operator+=(const ll a) { v = mod(v + a % MOD); }

    void operator+=(const signed a) { v = mod(v + a % MOD); }

    friend void operator+=(ll &a, const mint &b) { a = mod(a % MOD + b.v); }

    mint operator-(const mint &a) { return mint(v - a.v); }

    mint operator-(const ll a) { return mint(v - a % MOD); }

    mint operator-(const signed a) { return mint(v - a % MOD); }

    friend mint operator-(const ll a, const mint &b) { return mint(a % MOD - b.v); }

    void operator-=(const mint &a) { v = mod(v - a.v); }

    void operator-=(const ll a) { v = mod(v - a % MOD); }

    void operator-=(const signed a) { v = mod(v - a % MOD); }

    friend void operator-=(ll &a, const mint &b) { a = mod(a % MOD - b.v); }

    mint operator*(const mint &a) { return mint(v * a.v); }

    mint operator*(const ll a) { return mint(v * (a % MOD)); }

    mint operator*(const signed a) { return mint(v * (a % MOD)); }

    friend mint operator*(const ll a, const mint &b) { return mint(a % MOD * b.v); }

    void operator*=(const mint &a) { v = (v * a.v) % MOD; }

    void operator*=(const ll a) { v = mod(v * (a % MOD)); }

    void operator*=(const signed a) { v = mod(v * (a % MOD)); }

    friend void operator*=(ll &a, const mint &b) { a = mod(a % MOD * b.v); }

    mint operator/(const mint &a);

    mint operator/(const ll a);

    mint operator/(const signed a);

    friend mint operator/(const ll a, const mint &b);

    void operator/=(const mint &a);

    void operator/=(const ll a);

    void operator/=(const signed a);

    friend void operator/=(ll &a, const mint &b);


    //単項演算子
    mint operator+() { return *this; }

    mint operator++() {
        v++;
        return *this;
    }

    mint operator++(signed d) {
        mint res = *this;
        v++;
        return res;
    }

    mint operator-() { return operator*(-1); }

    mint operator--() {
        v++;
        return *this;
    }

    void operator--(signed d) {
        mint res = *this;
        v++;
    }

    bool operator==(mint &a) {
        return v == a.v;
    }

    bool operator==(signed a) {
        return v == a;
    }

    friend bool operator==(signed a, mint &b) {
        return a == b.v;
    }

    bool operator!=(mint &a) {
        return v != a.v;
    }

    bool operator!=(signed a) {
        return v != a;
    }

    friend bool operator!=(signed a, mint &b) {
        return a != b.v;
    }

    operator int() { return v; }
};

const int setModMax = 510000;
mint fac[setModMax], finv[setModMax], inv[setModMax];

void setMod() {
    fac[0] = fac[1] = 1;
    finv[0] = finv[1] = 1;
    inv[1] = 1;
    for (int i = 2; i < setModMax; i++) {
        fac[i] = fac[i - 1] * i % MOD;
        inv[i] = MOD - inv[MOD % i] * (MOD / i) % MOD;
        finv[i] = finv[i - 1] * inv[i] % MOD;
    }
}

mint minv(ll a) {
    if (a < setModMax) return inv[a];
    a %= MOD;
    ll b = MOD, x = 1, y = 0;
    while (b) {
        ll t = a / b;
        a -= t * b;
        swap(a, b);
        x -= t * y;
        swap(x, y);
    }
    return (x % MOD + MOD) % MOD;
}

mint mpow(mint &v, ll a) {
    ll x = v, n = a, res = 1;
    while (n) {
        if (n & 1)res = (res * x) % MOD;
        x = (x * x) % MOD;
        n >>= 1;
    }
    return res;
}


mint mint::operator/(const mint &a) { return mint(v * minv(a.v)); }

mint mint::operator/(const ll a) { return mint(v * minv(a)); }

mint mint::operator/(const signed a) { return mint(v * minv(a)); }

mint operator/(const ll a, const mint &b) { return mint(a % MOD * minv(b.v)); }

void mint::operator/=(const mint &a) { v = (v * minv(a.v)) % MOD; }

void mint::operator/=(const ll a) { v = mod(v * minv(a)); }

void mint::operator/=(const signed a) { v = mod(v * minv(a)); }

void operator/=(ll &a, const mint &b) { a = mint::mod(a % MOD * minv(b.v)); }


mint com(ll n, ll r) {
    if (n < r || n < 0 || r < 0)return 0;
    if (fac[0] == 0)setMod();
    return fac[n] * (finv[r] * finv[n - r] % MOD) % MOD;
}

ll u(ll a) {
    return a < 0 ? 0 : a;
}


#define ans(a) cout<<a<<endl;continue;

int N, M, H, W, cou, A[101010], B[101010], C[101010];
int dp[50][1010][1010];


signed main() {
    int a, b;
    cin >> N >> a >> b;
    for (int i = 0; i < N; i++) {
        cin >> A[i] >> B[i] >> C[i];
    }
    fill(dp, INF);
    dp[0][0][0] = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 1010; ++j) {
            for (int k = 0; k < 1010; ++k) {
                if (dp[i][j][k] == INF)continue;
                chmin(dp[i + 1][j][k], dp[i][j][k]);
                chmin(dp[i + 1][j + A[i]][k + B[i]], dp[i][j][k] + C[i]);

            }
        }
    }
    int res = INF;
    for (int i = 1; i < N + 1; ++i) {
        for (int j = 1; j < 1010; ++j) {
            for (int k = 1; k < 1010; ++k) {
                if ((ll) j * b == (ll) a * k)
                    chmin(res, dp[i][j][k]);
            }
        }
    }
    cout << (res == INF ? -1 : res) << endl;
    return 0;
}