/*** author: yuji9511 ***/
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using lpair = pair<ll, ll>;
const ll MOD = 1e9+7;
const ll INF = 1e18;
#define rep(i,m,n) for(ll i=(m);i<(n);i++)
#define rrep(i,m,n) for(ll i=(m);i>=(n);i--)
#define printa(x,n) for(ll i=0;i<n;i++){cout<<(x[i])<<" \n"[i==n-1];};
void print() {}
template <class H,class... T>
void print(H&& h, T&&... t){cout<<h<<" \n"[sizeof...(t)==0];print(forward<T>(t)...);}
vector<ll> tree[100010];
ll num[100010] = {};
int main(){
    cin.tie(0);
    ios::sync_with_stdio(false);
    ll N;
    cin >> N;
    ll A[100010], B[100010];
    rep(i,0,N-1){
        cin >> A[i] >> B[i];
        A[i]--; B[i]--;
        tree[A[i]].push_back(B[i]);
        tree[B[i]].push_back(A[i]);
    }
    rep(i,0,N) num[i] = -1;
    ll K;
    cin >> K;
    ll V[100010], P[100010];
    rep(i,0,K){
        cin >> V[i] >> P[i];
        V[i]--;
        num[V[i]] = P[i];
    }
    priority_queue<lpair, vector<lpair>, greater<lpair> > pq;
    rep(i,0,N){
        if(num[i] != -1) pq.push({num[i], i});
    }
    bool ok = true;
    while(not pq.empty()){
        lpair l1 = pq.top();
        pq.pop();
        ll idx = l1.second;
        for(auto &e: tree[idx]){
            if(num[e] == -1){
                num[e] = num[idx] + 1;
                pq.push({num[e], e});
            }else{
                if(abs(num[e] - num[idx]) != 1) ok = false;
            }
        }
    }
    if(ok){
        print("Yes");
        rep(i,0,N) print(num[i]);
    }else{
        print("No");
    }
    
    

}