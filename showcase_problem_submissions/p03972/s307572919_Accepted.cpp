#include<bits/stdc++.h>

#define rep(i,n) for(auto i=0*(n);i<(n);++i)

typedef long long ll;

using namespace std;

struct UF {
  int groups;
  vector<int> parent;
  UF(int n):parent(n,-1),groups(n){}
  int root(int x){return parent[x]<0?x:parent[x]=root(parent[x]);}
  bool merge(int x,int y){
    x=root(x);y=root(y);
    if(x==y)return false;
    if(parent[y]<parent[x])swap(x,y);
    if(parent[x]==parent[y])--parent[x];
    parent[y]=x;
    --groups;
    return true;
  }
};

int main(){
  int W, H;
  cin>>W>>H;

  vector<int> cx(W),cy(H);
  rep(i,W)cin>>cx[i];
  rep(i,H)cin>>cy[i];

  vector<pair<ll,pair<int,int>>> cc;
  rep(i,W)cc.push_back({cx[i],{0,i}});
  rep(i,H)cc.push_back({cy[i],{1,i}});

  sort(cc.begin(),cc.end());

  ll gx = W + 1;
  ll gy = H + 1;
  ll cost=0;

  UF ux(W+1),uy(H+1);

  rep(i,cc.size()){
    if(cc[i].second.first == 0){
      --gx;
      cost += gy * cc[i].first;
    }else{
      --gy;
      cost += gx * cc[i].first;
    }
  }

  cout << cost << endl;

  return 0;
}
