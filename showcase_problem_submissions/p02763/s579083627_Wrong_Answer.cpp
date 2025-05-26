#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef vector<ll> vl;
typedef vector<vl> vvl;
typedef pair<ll,ll> pl;
typedef vector<pl> vp;
#define fore(i,a,b) for(ll i=(ll)(a);i<=(ll)(b);++i)
#define rep(i,n) fore(i,0,(n)-1)
#define rfore(i,a,b) for(ll i=(ll)(b);i>=(ll)(a);--i)
#define rrep(i,n) rfore(i,0,(n)-1)
#define all(x) (x).begin(),(x).end()
const ll INF=1001001001;
const ll LINF=1001001001001001001;
const ll D4[]={0,1,0,-1,0};
const ll D8[]={0,1,1,0,-1,-1,1,-1,0};
template<class T>
bool chmax(T &a,const T &b){if(a<b){a=b;return 1;}return 0;}
template<class T>
bool chmin(T &a,const T &b){if(b<a){a=b;return 1;}return 0;}

template< typename T >
struct BinaryIndexedTree {
  vector< T > data;

  BinaryIndexedTree(int sz) {
    data.assign(++sz, 0);
  }

  T sum(int k) {
    T ret = 0;
    for(++k; k > 0; k -= k & -k) ret += data[k];
    return (ret);
  }

  void add(int k, T x) {
    for(++k; k < data.size(); k += k & -k) data[k] += x;
  }
};

void solve(){
    ll n;cin>>n;
    string s;cin>>s;
    ll q;cin>>q;
    vector<BinaryIndexedTree<ll>> g(26,BinaryIndexedTree<ll>(n+1));
    rep(i,n){
        ll idx=s[i]-'a';
        g[idx].add(i,1);
    }
    while(q--){
        ll t;cin>>t;
        if(t==1){
            ll i;cin>>i;
            i--;
            char c;cin>>c;
            ll idx=c-'a';
            ll idx2=s[i]-'a';
            g[idx2].add(i,-1);
            g[idx].add(i,1);
        }else{
            ll l,r;cin>>l>>r;
            l--,r--;
            ll cnt=0;
            rep(i,26){
                ll tmp=g[i].sum(r);
                if(l>0)tmp-=g[i].sum(l-1);
                if(tmp>0)cnt++;
            }
            cout<<cnt<<endl;
        }
    }
}

int main(){
    cin.tie(0);
    ios::sync_with_stdio(0);
    solve();
}
