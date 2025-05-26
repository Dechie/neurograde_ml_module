#include <iostream>
#include <cstdio>
#include <algorithm>
#include <string>
#include <cstring>
#include <cctype>
#include <cmath>
#include <stack>
#include <queue>
#include <vector>
#include <set>
#include <map>
#include <list>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <math.h>
#include <bitset>
#include <iterator>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <cassert>
#define INF 200000000000000	
#define MOD 1000000007
#define sym cout<<"---------"<<endl;
#define ll long long
#define mk make_pair
#define en endl
#define RE return 0
#define int ll
#define P pair<int,int>
using namespace std;
int dx[5]={1,0,-1,0,0},dy[5]={0,1,0,-1,0};
int gcd(int a,int b){if(a%b==0){return b;}else return gcd(b,a%b);}
int lcm(int a,int b){if(a==0){return b;} return a/gcd(a,b)*b;}

pair<P, int> G[45] ; // 物質の情報

int dp[45][1050][1050];

signed main(){
	int n,m1,m2; cin>>n>>m1>>m2;
	int sum_a=0,sum_b=0,s=0;
	for(int i=0; i<n; i++){
		int c,d,e; cin>>c>>d>>e;
		sum_a+=c; sum_b+=d; s+=e;
		G[i]=mk(mk(c,d), e);
	}
	
	for(int k=0; k<=43; k++) for(int i=0; i<=1040; i++) for(int j=0; j<=1040; j++) dp[k][i][j]=INF+10000;
	
	dp[0][0][0]=0;
	for(int i=0; i<n; i++){
		int a=G[i].first.first,b=G[i].first.second,c=G[i].second;
		for(int j=a; j<=sum_a; j++){
			for(int k=b; k<=sum_b; k++){
				dp[i+1][j][k]=min(dp[i+1][j][k], dp[i][j-a][k-b]+c);
				dp[i+1][j][k]=min(dp[i+1][j][k], dp[i][j][k]);
			}
		}
	}
	int ans=INF;
	for(int i=1; i<=sum_a; i++){
		for(int j=1; j<=sum_b; j++){
			if(i*m2==j*m1){
				ans=min(ans, dp[n][i][j]);
			}
		}
	}
	if(ans<INF) cout<<ans<<en;
	else cout<<-1<<en;
}

