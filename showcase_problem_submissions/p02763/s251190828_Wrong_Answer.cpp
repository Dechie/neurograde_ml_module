#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <cstdio>
#include <string>
#include <cmath>
#include <queue>
#include <tuple>
#include <set>
#include <assert.h>
#include <sstream>
#include <string>
//#include <bits/stdc++.h>
#define maxs(x,y) x = max(x,y)
#define mins(x,y) x = min(x,y)
#define rep(i,n) for(int (i)=0;(i)<(n);(i)++)
#define repr(i, n) for (int i = (n) - 1; i >= 0; i--)
#define FOR(i,i0,n) for(int (i)=(i0);(i)<(n);(i)++)
#define FORR(i,i0,n) for(int (i)=(n)-1; (i)>=(i0);(i)--)
#define SORT(x) sort(x.begin(),x.end())
#define SORTR(x) sort(x.begin(),x.end(),greater <>())
#define rn return
#define fi first
#define se second
#define pb push_back
#define eb emplace_back
#define mp make_pair
#define mt make_tuple
using namespace std;
using ll = long long;
typedef std::pair<ll,ll> P;
//#include <boost/multiprecision/cpp_int.hpp>
//using bint = boost::multiprecision::cpp_int;


int main(){
    int n;
    string s;
    int q;
    cin >> n >> s >> q;
    vector<set<int>> archive(26);
    rep(i,n){
        archive[s[i]-'a'].insert(i);
    }
    vector<int> ans;
    rep(qi,q){
        int t;
        cin >> t;
        if (t==1){
            int l; cin >> l;
            l--;
            char c; cin >> c;
            if (c!= s[l]){
                archive[s[l] - 'a'].erase(l);
                archive[c-'a'].insert(l);
            }
        }
        else{
            int l,r; cin >> l >> r;
            l--;r--;
            int count = 0;
            rep(j,26){
                auto lb = archive[j].lower_bound(l);
                if (lb != archive[j].end() && *lb<=r) count++;
            }
            ans.pb(count);
        }
    }
    rep(i,ans.size()) cout << ans[i] << endl;
    rn 0;
}
