#include <iostream>
#include <bits/stdc++.h>

using namespace std;
int n,q,len;
int D[100005],dep[100005],dis[100005];
int anc[100005][30];
int fel[100005],ccnt[100005],csums[100005];
int eul[200005];
vector<pair<int,int>> ct[100005];
vector<int> csum[100005];

struct it{
    int f,s,t;
    it(int ff,int ss,int tt){
        f=ff;s=ss;t=tt;
    }
};

vector<it> rad[100005];
vector<pair<int,int>> cr[100005];

void dfs(int now,int from,int ds)
{

    eul[++len] = now;
    fel[now] = len;
    dis[now] = ds;


    if(from!=-1)
    {dep[now] = dep[from]+1;anc[now][0] = from;}
    for(int i=1;i<20;++i)
    {
        anc[now][i] = anc[anc[now][i-1]][i-1];
    }

    for(int i=0;i<rad[now].size();++i)
    {
        if(rad[now][i].f==from) continue;
        ct[rad[now][i].s].push_back(pair<int,int>(len,++ccnt[rad[now][i].s]));
        csum[rad[now][i].s].push_back(csums[rad[now][i].s]+=rad[now][i].t);
        dfs(rad[now][i].f,now,ds+rad[now][i].t);
        eul[++len] = now;
        ct[rad[now][i].s].push_back(pair<int,int>(len,--ccnt[rad[now][i].s]));
        csum[rad[now][i].s].push_back(csums[rad[now][i].s]-=rad[now][i].t);
    }
}

int lca(int u,int v)
{
    if(dep[u]<dep[v])swap(u,v);
    int tmp = dep[u] - dep[v];
    for(int j=0;tmp;++j,tmp>>=1)
        if(tmp&1) u = anc[u][j];

    for(int i=20;i>=0&&u!=v;--i)
    {
        if(anc[u][i]!=anc[v][i])
        {
            u = anc[u][i];
            v = anc[v][i];
        }
    }
    return anc[u][0];
}

int gcost(int u,int x,int y)
{
    int pos = lower_bound(ct[x].begin(),ct[x].end(),pair<int,int>(fel[u],0)) - ct[x].begin();
    if(pos==ct[x].size()||ct[x][pos].first>fel[u]) --pos;
    //cout << pos << " "<<csum[x][pos] << " " << ct[x][pos].second*y << endl;
    return dis[u] - csum[x][pos] + ct[x][pos].second*y;
}

int main()
{
    scanf("%d%d",&n,&q);
    int a,b,c,d;
    for(int i=1;i<n;++i)
    {
        scanf("%d%d%d%d",&a,&b,&c,&d);
        rad[a].push_back(it(b,c,d));
        rad[b].push_back(it(a,c,d));
        D[c] = d;
        ct[i].push_back(pair<int,int>(1,0));
        csum[i].push_back(0);
    }
    memset(fel,-1,sizeof(fel));
    anc[1][0]=1;
    dfs(1,-1,0);

    int x,y,u,v;
    for(int i=0;i<q;++i)
    {
        scanf("%d%d%d%d",&x,&y,&u,&v);
        int l = lca(u,v);

        printf("%d\n",gcost(u,x,y)+gcost(v,x,y)-2*gcost(l,x,y));
    }
    return 0;
}
