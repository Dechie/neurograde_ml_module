#include <iostream>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <utility>
#include <string>
#include <cmath>
#include <set>
#include <functional>
#include <map>
#include <queue>
#include <cstring>
#include <stack>
#include <iomanip>
#include <climits>
#include <numeric>
#include <cassert>
#include <unordered_map>
using namespace std;
#define rep(i,n) for(int i = 0; i < (n); ++i)
#define show(x) cout << #x << " = " << x << endl;
typedef long long ll;
typedef pair<int,int> pii;

void solve2(){
    ll n;
    cin >> n;
    ll a[50];
    rep(i,n){
        cin >> a[i];
    }
    ll ans = 0;
    while(1){
        ll sum = 0LL;
        rep(i,n){
            sum += a[i]/n;
        }
        rep(i,n){
            ll k = a[i]/n;
            a[i] -= k*n;
            a[i] += sum-k;
        }
        ans += sum;
        if(sum==0LL){
            cout << ans << endl;
            exit(0);
        }
    }
    
}
void solve1(){
    ll k;
    cin >> k;
    int n = 50;
    vector<ll>a(n);
    int c = 0;
    while(k%50)k+=51,c++;
    //show(k);
    ll m = k/50LL+49;
    rep(i,n){
        if(i < c)a[i] = m-i-50-1;
        else a[i] = m;
    }
    cout << n << endl;
    rep(i,n-1){
        cout << a[i] << " ";
    }
    cout << a[n-1] << endl;
}

int main(){
    solve2();
}