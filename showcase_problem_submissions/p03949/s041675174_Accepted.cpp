#include<iostream>
#include<algorithm>
#include<vector>
#include<queue>
#include<set>
#include<unordered_map>
using namespace std;
typedef long long ll;
#define chmax(a,b) a=max(a,b)
#define chmin(a,b) a=min(a,b)
#define mod 1000000007
#define mad(a,b) a=(a+b)%mod
#define N 200010
ll n,k;
ll p[N];
vector<ll> g[N];
ll l[N],r[N];
void dfs1(ll x,ll from){
    for(auto y:g[x])if(y!=from){
	dfs1(y,x);
    }
    if(-1!=p[x]){
	l[x]=r[x]=p[x];
    }
    else{
	l[x]=-1e17,r[x]=1e17;
	for(auto y:g[x])if(y!=from){
	    chmax(l[x],l[y]-1);
	    chmin(r[x],r[y]+1);
	}
    }
}
bool dfs2(ll x,ll from,ll t){
    //if((t+1)%2!=l[x]%2&&l[x]>-1e9)return 0;
    if(l[x]<=t+1&&t+1<=r[x])p[x]=t+1;
    else if(l[x]<=t-1&&t-1<=r[x])p[x]=t-1;
    else return 0;
    for(auto y:g[x])if(y!=from){
	if(dfs2(y,x,p[x])==0)return 0;
    }
    return 1;
}
int main(){
    cin.tie(0);
    ios::sync_with_stdio(0);
    cin>>n;
    for(int i=0;i<n-1;i++){
	ll a,b; cin>>a>>b;
	g[a].push_back(b);
	g[b].push_back(a);
    }
    for(int i=1;i<=n;i++){
	p[i]=-1;
    }
    cin>>k;
    ll root=-1;
    for(int i=0;i<k;i++){
	ll v; cin>>v;
	cin>>p[v];
	root=v;
    }
    dfs1(root,0);
    /*cout<<root<<endl;
    for(int i=1;i<=n;i++)cout<<i<<":"<<l[i]<<" "<<r[i]<<endl;*/
    bool ok=1;
    for(auto x:g[root]){
	ok&=dfs2(x,root,p[root]);
    }
    if(ok){
	cout<<"Yes"<<endl;
	for(int i=1;i<=n;i++){
	    for(auto x:g[i]){
		if(abs(p[i]-p[x])!=1)cout<<1/0<<endl;
	    }
	    cout<<p[i]<<endl;
	}
    }
    else cout<<"No"<<endl;
}


