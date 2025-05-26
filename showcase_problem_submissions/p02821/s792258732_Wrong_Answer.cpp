//#define _GLIBCXX_DEBUG
#include <bits/stdc++.h>
using namespace std;
#define repd(i,a,b) for (int i=(a);i<(b);i++)
#define rep(i,n) repd(i,0,n)
typedef long long ll;
typedef long double lb;
#define int long long
#define double long double
#define endn "\n"
typedef pair<int,int> P;
template<class T> inline bool chmin(T& a, T b){if(a>b){a = b;return 1;}return 0;}
template<class T> inline bool chmax(T& a, T b){if(a<b){a = b;return 1;}return 0;}
const int MOD = 1000000007;
//const int MOD = 998244353;
template<class T> inline int add(T& a, T b, T M = MOD){a=(a+M)%M;b=(b+M)%M;a=(a+b)%M;return a;};
template<class T> inline int mul(T& a, T b, T M = MOD){if(a>=M)a%=M;if(b>=M)b%=M;a*=b;if(a>=M)a%=M;return a;};
const ll INF = 1e16;
const double EPS = 1e-10;
const double PI = 3.141592653589793;
const string abc="abcdefghijklmnopqrstuvwxyz";
const string ABC="ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const pair<int,int> fd[] = {make_pair(1,0),make_pair(-1,0),make_pair(0,1),make_pair(0,-1)}; 

////////////////////////////////////////////////////////////////////
///////////////////////___modpow___////////////////////
ll modpow(ll a,  ll n, const ll &MOD = MOD){
  ll ret = n == 0 ? 1 : modpow(a, n/2, MOD);
  (ret *= ret) %= MOD;
  if(n%2)((ret *= a) %= MOD);
  return ret;
}
///////////////////////___modinv___////////////////////
ll modinv(ll d, const ll &MOD = MOD){
  return modpow(d, MOD-2, MOD);
}
////////////////////////////////////////////////////////////////////

int n,m,a[110000],r[110000];
priority_queue<int> que;
signed main(){
  cin>>n>>m;rep(i,n)cin>>a[i];
  sort(a,a+n,greater<int>());
  r[0] = a[0];
  repd(i,1,n)r[i] = a[i]+r[i-1];
  int ng = INF, ok = 0, ans = 0;
  while(abs(ng-ok) > 1){
    int mid = (ok+ng)/2;
    int cnt = 0, sum = 0, minus = INF;
    rep(i,n){
      auto itr = upper_bound(a, a+n, mid-a[i], greater<int>());
      if(itr != a){
        int ind = itr-a;
        cnt += ind;
        sum += r[ind-1]+a[i]*ind;
        chmin(minus, a[ind-1]+a[i]);
      }
    }
    if(cnt > m)sum -= minus;
    (cnt >= m ? ok : ng) = mid;
    ans = (cnt >= m ? sum : ans);
  }
  cout << ans << endl;
}
