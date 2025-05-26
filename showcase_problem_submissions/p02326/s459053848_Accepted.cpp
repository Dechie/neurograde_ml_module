#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <queue>
#include <stack>
#include <map>
#include <set>
#include <string>
#include <cmath>
using namespace std;
#define MOD 1000000007
#define INF 1<<30
#define LINF (ll)1<<62
#define rep(i,n) for(ll i=0; i<(n); i++)
#define REP(i,a,b) for(ll i=(a); i<(b); i++)
#define all(x) (x).begin(),(x).end()
using namespace std;
typedef long long ll;
typedef vector<ll> vl;
typedef vector<vl> vvl;
typedef pair<ll, ll> P;
typedef vector<pair<ll, ll>> vpl;

ll min3(ll a, ll b, ll c){
    if(a < b){
        if(b < c) return a;
        else{
            if(a < c) return a;
            else return c;
        }
    }else{
        if(b < c) return b;
        else return c;
    }
}

int main(){
    ll h,w; cin >> h >> w;
    ll c[h][w];
    rep(i,h){
        rep(j,w){
            cin >> c[i][j];
        }
    }
    ll dp[h+1][w+1] = {};
    rep(i,h){
        rep(j,w){
            if(c[i][j] == 1) dp[i+1][j+1] == 0;
            else{
                dp[i+1][j+1] = min3(dp[i][j],dp[i][j+1],dp[i+1][j]) + 1;
            }
        }
    }
    ll mx = 0;
    rep(i,h+1){
        rep(j,w+1){
            if(mx < dp[i][j]) mx = dp[i][j];
        }
    }
    cout << mx*mx << endl;
}
