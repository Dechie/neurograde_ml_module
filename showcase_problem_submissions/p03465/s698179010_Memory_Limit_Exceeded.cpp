//  DO NOT USE:  A   N (segtree)

#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int,int>
#define f(a,b,c) for(int a=b;a<c;a++)
#define read(t) ll t; cin>>t;
#define readarr(arr,n) ll arr[n]; f(i,0,n) cin>>arr[i];
#define forstl(i,v) for(auto &i: v)
#define ln endl
#define dbg(x) cout<<#x<<" = "<<x<<ln;
#define dbg2(x,y) cout<<#x<<" = "<<x<<" & "<<#y<<" = "<<y<<ln;
#define dbgstl(v) cout<<#v<<" = \n"; { int c=0; forstl(it,v) \
cout<<"    Term "<< ++c <<" = "<<it<<ln;} cout<<ln;
#define dbgstlp(v) cout<<#v<<" = \n"; { int c=0; forstl(it,v) \
cout<<"    Term "<< ++c <<" = "<<it.fi<<" , "<<it.se<<ln;} cout<<ln;
#define dbgarr(v,s,e) cout<<#v<<" = "; f(i,s,e) cout<<v[i]<<", "; cout<<ln;
#define addEdge(tr,k) f(i,0,k) { ll x,y; cin>>x>>y, tr[x].push_back(y), tr[y].push_back(x);}

mt19937_64 rang(chrono::high_resolution_clock::now().time_since_epoch().count());
int rng(int lim) { uniform_int_distribution<int> uid(0, lim - 1); return uid(rang); }


//														GLOBAL VARS
ll INF =LLONG_MAX;
const ll M= 1000000007;

//														OBJECTS
struct obj
{
    ll x,y;
};

//														USEFUL FUNCTIONS
ll powm(ll , ll ) ;

//          											SEGMENT TREE(!N)

const int N=3e0+5;
ll tree[4*N+1];
ll A[N];
ll lazy[4*N+1]={0};

void build(ll,ll,ll);
void update(ll,ll,ll,ll,ll);
ll query(ll,ll,ll,ll,ll);
void updateRange(ll,ll,ll,ll,ll,ll);
ll queryRange(ll,ll,ll,ll,ll);

bitset<4000000>b[2000];

// 														MAIN
int main()
{ 
	ios_base::sync_with_stdio(false);
    cin.tie(NULL);
	cout.tie(NULL);
    //It gets easier day by day, but the hard part is to........
	read(n)
    readarr(a,n)
    sort(a,a+n);
    ll sum=0;
    f(i,0,n)
    {
        sum+=a[i];
    }
    b[0].set(0,1);
    b[0].set(a[0],1);
    f(i,1,n)
    {
        b[i]=b[i]|b[i-1];
        b[i]=b[i]|(b[i-1]<<a[i]);
    }
    sum++;
    sum/=2;
    f(i,sum,2*sum+5)
    if(b[n-1][i]) {cout<<i<<endl;return 0;}
    
}










//													FUNCTIONS DECLARATIONS
ll powm(ll a, ll b) {
	ll res=1;
	while(b) {
		if(b&1)
			res=(res*a)%M;
		a=(a*a)%M;
		b>>=1;
	}
	return res;
}

void build(ll node, ll start, ll end)
{
    if(start == end)
    {
        // Leaf node will have a single element
        tree[node] = A[start];
    }
    else
    {
        ll mid = (start + end) / 2;
        // Recurse on the left child
        build(2*node, start, mid);
        // Recurse on the right child
        build(2*node+1, mid+1, end);
        // llernal node will have the sum of both of its children
        tree[node] = tree[2*node] + tree[2*node+1];
    }
}
void update(ll node, ll start, ll end, ll idx, ll val)
{
    if(start == end)
    {
        // Leaf node
        A[idx] += val;
        tree[node] += val;
    }
    else
    {
        ll mid = (start + end) / 2;
        if(start <= idx and idx <= mid)
        {
            // If idx is in the left child, recurse on the left child
            update(2*node, start, mid, idx, val);
        }
        else
        {
            // if idx is in the right child, recurse on the right child
            update(2*node+1, mid+1, end, idx, val);
        }
        // llernal node will have the sum of both of its children
        tree[node] = tree[2*node] + tree[2*node+1];
    }
}

ll query(ll node, ll start, ll end, ll l, ll r)
{
    if(r < start or end < l)
    {
        // range represented by a node is completely outside the given range
        return 0;
    }
    if(l <= start and end <= r)
    {
        // range represented by a node is completely inside the given range
        return tree[node];
    }
    // range represented by a node is partially inside and partially outside the given range
    ll mid = (start + end) / 2;
    ll p1 = query(2*node, start, mid, l, r);
    ll p2 = query(2*node+1, mid+1, end, l, r);
    return (p1 + p2);
}

void updateRange(ll node, ll start, ll end, ll l, ll r, ll val)
{
    if(lazy[node] != 0)
    { 
        // This node needs to be updated
        tree[node] += (end - start + 1) * lazy[node];    // Update it
        if(start != end)
        {
            lazy[node*2] += lazy[node];                  // Mark child as lazy
            lazy[node*2+1] += lazy[node];                // Mark child as lazy
        }
        lazy[node] = 0;                                  // Reset it
    }
    if(start > end or start > r or end < l)              // Current segment is not within range [l, r]
        return;
    if(start >= l and end <= r)
    {
        // Segment is fully within range
        tree[node] += (end - start + 1) * val;
        if(start != end)
        {
            // Not leaf node
            lazy[node*2] += val;
            lazy[node*2+1] += val;
        }
        return;
    }
    ll mid = (start + end) / 2;
    updateRange(node*2, start, mid, l, r, val);        // Updating left child
    updateRange(node*2 + 1, mid + 1, end, l, r, val);   // Updating right child
    tree[node] = tree[node*2] + tree[node*2+1];        // Updating root with max value 
}

ll queryRange(ll node, ll start, ll end, ll l, ll r)
{
    if(start > end or start > r or end < l)
        return 0;         // Out of range
    if(lazy[node] != 0)
    {
        // This node needs to be updated
        tree[node] += (end - start + 1) * lazy[node];            // Update it
        if(start != end)
        {
            lazy[node*2] += lazy[node];         // Mark child as lazy
            lazy[node*2+1] += lazy[node];    // Mark child as lazy
        }
        lazy[node] = 0;                 // Reset it
    }
    if(start >= l and end <= r)             // Current segment is totally within range [l, r]
        return tree[node];
    ll mid = (start + end) / 2;
    ll p1 = queryRange(node*2, start, mid, l, r);         // Query left child
    ll p2 = queryRange(node*2 + 1, mid + 1, end, l, r); // Query right child
    return (p1 + p2);
}
