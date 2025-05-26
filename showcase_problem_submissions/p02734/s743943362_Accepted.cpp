#include<bits/stdc++.h>
using namespace std;
using ll = long long;
#define int long long
#ifdef LOCAL
#include<debug.h>
#else
#define db(...) 21
#endif
#define sz(a) (int)a.size()
#define found(m,x) ((m).find(x)!=(m).end())
#define uset unordered_set
#define umap unordered_map
#define a(x) x.begin(),x.end()
#define fo(i,n) for(int i=0;i<(int)n;i++)
#define Fo(i,k,n) for(int i=k;k<n?i<=(int)n:i>=(int)n;k<n?i++:i--)
#define input(v,n) fo(i,n)cin>>v[i]
using vi = vector<int>;
using vpi = vector<pair<int,int>>;
using pi = pair<int,int>;

const int nax = 1e5+7;
const int mod = 998244353; //1e9+2667 :: 0xcf : inf, 0xc0 -inf
const int inf = numeric_limits<int> :: max() - (int)9e6;
// ===================== MANASH =================== //

int n;

void test() {

    cin >> n;
    int sum;
    cin >> sum;
    vi a;
    a.resize(n);
    input(a,n);
    int dp[n+1][sum+1][3];
    memset(dp,0,sizeof(dp));
    dp[0][0][0] = 1;

    for( int i = 0; i < n; i++) {
        for( int s = 0; s <= sum; s++) {

            (dp[i+1][s][0] += dp[i][s][0]) %= mod;
            (dp[i+1][s][1] += dp[i][s][0] + dp[i][s][1]) %= mod;
            (dp[i+1][s][2] += dp[i][s][0] + dp[i][s][2] + dp[i][s][1]) %= mod;
            if ( s + a[i] > sum ) continue;
            (dp[i+1][s+a[i]][2] += dp[i][s][1] + dp[i][s][0]) %= mod;
            (dp[i+1][s+a[i]][1] += dp[i][s][0] + dp[i][s][1]) %= mod;
        }
    }
    cout << dp[n][sum][2];
}

signed main() {

    ios_base :: sync_with_stdio(false);
    cin.tie(NULL);

    #ifdef LOCAL
    freopen("input.txt","r",stdin);
    #endif

    int T = 1;
    //cin >> T;
    while(T--) {
        test();
        cout << "\n";
    }
    return 0;

}