#include <cstdio>
#include <algorithm>
#include <iostream>
#include <vector>
#define MAX_N 100000100

using namespace std;

int par[100100];
int rank1[100100];

void init(int n){
  int i;
  for(i=0;i<n;i++){
    par[i]=i;
    rank1[i]=0;
  }
}
int find(int x){
  if(par[x]==x){
    return x;
  }
  else {
    return par[x]=find(par[x]);
  }
}

void unite(int x ,int y){
  x=find(x);
  y=find(y);
  if(x==y) return ;

  if((int)rank1[x]<(int)rank1[y]){
    par[x]=y;
  }else {
    par[y]=x;
    if(rank1[x]==rank1[y]) rank1[x]++;
  }
}
bool same(int x,int y)
{
  return find(x)==find(y);
}

struct edge{int u,v,cost;};

bool comp(const edge& e1,const edge& e2){
  return e1.cost<e2.cost;
}

edge es[MAX_N];
int V,E;

int kruskal(){
  sort(es,es+E,comp);
  init(V);
  int res=0;
  for(int i=0;i<E;i++){
    edge e=es[i];
    if(!same(e.u,e.v)){
      unite(e.u,e.v);
      res += e.cost;
    }
  }
  return res;
}

int main(void){
  string S;
  int W,H;
  int res;
  cin>>W>>H;
  int i,j;
  vector<int> p(W);
  vector<int> q(H);
  int k=0;

  for(i=0;i<W;i++) cin>>p[i];
  for(i=0;i<H;i++) cin>>q[i];

  V=(W+1)*(W+1);
  E=((W+1)*H)+(W*(H+1));

  k=0;
  for(j=0;j<=H;j++){
    for(i=0;i<=W;i++) {
      if(i<W){
        es[k].u=j*(W+1)+i;      
        es[k].v=j*(W+1)+i+1;
        es[k].cost=p[i];
        k++;
      }
      if(j<H){
        es[k].u=j*(W+1)+i;      
        es[k].v=j*(W+1)+i+W+1;
        es[k].cost=q[j];
        k++;
      }
    }
  }
  res=kruskal();
  cout<<res<<endl;
  return 0;
}