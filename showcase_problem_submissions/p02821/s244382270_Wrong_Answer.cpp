#include <bits/stdc++.h>
#define repr(i,from,to) for(int (i)=(from);(i)<(to);(i)++)
#define rep(i,n) for(int (i)=0;(i)<(n);(i)++)
// const bool debug=true;
const bool debug=false;
#define DEBUG if(debug==true)
#define vprint(x) for(auto a:(x)) cout << x << endl;
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
const ll MOD = 1000000007;
template<class T> inline bool chmin(T& a, T b) {
    if (a > b) { a = b; return true;}
    return false;
}
template<class T> inline bool chmax(T& a, T b) {
    if (a < b) { a = b; return true;}
    return false;
}
int main(void) {
    cin.tie(0);
    ios::sync_with_stdio(false);
    ll n,m;
    cin>>n>>m;
    vector<ll> a(n);rep(i,n)cin>>a[i];
    sort(begin(a), end(a));
    function<bool(ll)> func = [&] (ll value) {
        ll cnt = 0;
        rep(i,n) {
            auto ptr = lower_bound(begin(a), end(a), value-a[i]);
            cnt += ll(end(a)-ptr);
        }
        return cnt >= m;
    };
    ll from = 0,to = 1e+5;
    while(to>from) {
        ll mid = (from + to) / 2;
        if (func(mid)) {
            from = mid;
        }
        else {
            to = mid;
        }
        if (to==from+1) {
            if (func(to)) {
                from = to;
            }
            else { 
                to = from;
            }
        }
    }
    vector<ll> csum(n+1,0);
    rep(i,n) {
        csum[i+1] = csum[i] + a[i];
    }  
    ll res = 0;
    ll cnt = 0;
    rep(i,n) {
        auto ptr = lower_bound(begin(a), end(a), from+1-a[i]);
        int idx = int(ptr-a.begin());
        res += csum[n] - csum[idx] + a[i] * (n-idx);
        cnt += n-idx;
    }
    res += from * (m-cnt);
    cout << res << endl;
    return 0;
}
