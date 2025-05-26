#include <iostream>
using namespace std;

int r[101][101];
int d[101], f[101], num, visit[100]={0}, p=0;

void dfs(int n){
    if(visit[n]!=1){
        visit[n] = 1;
        d[n] = ++p;
        for(int i=1; i<num+1; i++)
            if(r[n][i]==1) dfs(i);
        f[n] = ++p;
    }
}

int main(void){
    int u, k, v;
    cin >> num;
    for(int i=0; i<num; i++){
        cin >> u >> k;
        for(int j=0; j<k; j++){
            cin >> v;
            r[u][v]=1;
        }
    }
    for(int i=1; i<num+1; i++){
        if(visit[i]!=1) dfs(i);
    }
    for(int i=1; i<num+1; i++)
        cout << i << " " << d[i] << " " << f[i] << endl;
    return 0;
}

