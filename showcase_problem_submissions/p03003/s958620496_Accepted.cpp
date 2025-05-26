#include<bits/stdc++.h>
using namespace std;
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;

#pragma comment(linker, "/stack:200000000")
#pragma GCC optimize("Ofast")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")

#define start_routine() int begtime = clock();
#define end_routine() int endtime = clock(); cerr << endl << "Time elapsed: " << (endtime - begtime)*1000/CLOCKS_PER_SEC << " ms"; return 0
#define speed() cin.tie(0), cout.tie(0), ios_base::sync_with_stdio(false)
// #define exit(a, b) return cout << a, b;

#define loop(i,a,b) for(ll i=a;i<b;i++)
#define all(v) v.begin(), v.end() 

#define print(stuff) cout << stuff << endl
#define printc(stuff) for(auto x: stuff) cout << x << " "; cout << endl;
#define len length
#define ret0 return 0
#define ret return 

#define ll long long
#define ld long double
#define fi first
#define endl '\n'
#define se second
#define pb push_back
#define mp make_pair
#define lb lower_bound
#define ub upper_bound

#define vl vector<ll> 
#define sl set<ll>
#define pll pair<ll, ll>
#define mll map<ll, ll> 
#define pq priority_queue<ll>

// typedef tree<ll,null_type,less<ll>,rb_tree_tag,
// tree_order_statistics_node_update> indexed_set;

#define inf (long long int) 1e18
#define eps 0.000001
#define mod 1000000007
#define mod1 998244353
#define MAXN 2005

ll n, m;
ll s[MAXN], t[MAXN];
ll dp[MAXN][MAXN];

ll func(ll i, ll j){
    if(i == n || j == m){
        ret0;
    }
    else if(dp[i][j] !=-1) ret dp[i][j];
    else {
        if(s[i] == t[j]){
            return dp[i][j] =  (1 + (func(i+1, j)%mod + func(i, j+1)%mod)%mod)%mod;
        }
        else if(s[i] != t[j]){
            ret dp[i][j] = ((func(i+1, j)%mod + func(i, j+1)%mod)%mod - func(i+1, j+1)%mod + mod)%mod;
        }
    }
}

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    

    cin>>n>>m;
    loop(i,0,n) cin>>s[i];
    loop(i,0,m) cin>>t[i];
    loop(i,0,MAXN) loop(j,0,MAXN) dp[i][j] = -1;
    print(func(0, 0) + 1);
   
}

