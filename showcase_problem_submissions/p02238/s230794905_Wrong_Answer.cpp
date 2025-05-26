#if 0

#endif

#include <iostream>
#include <vector>
using namespace std;

int n;
int t; //mytime?????????????????§???????´?
int mytime1[110],mytime2[110]; //?????§??????
int G[110][110];
int c[110];

void dfs(int cur){
	t+=1;
	mytime1[cur]=t;
	c[cur]=1;
	for(int dst=0; dst<n; ++dst){
		if(G[cur][dst]==1&&c[dst]==0){
			cout<<1<<endl;
			dfs(dst);
		}
	} 
	t += 1;
	mytime2[cur]=t;
}


int main()
{
	int G[110][110];
	cin>>n;
	for(int i=0;i<n;i++){
		int u,k;
		cin>>u>>k;
		for(int j=0;j<k;j++){
			int tmp;
			cin>>tmp;
			G[u][tmp]=1;
		}
	}
	
	for(int i=0;i<n;++i){
		dfs(i);
		cout<<(i+1)<<' '<<mytime1[i+1]<<" "<<mytime2[i+1] <<endl;
	}
}