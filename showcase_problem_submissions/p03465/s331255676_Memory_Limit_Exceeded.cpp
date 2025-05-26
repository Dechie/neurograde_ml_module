#include <bits/stdc++.h>
#define MOD 1000000007LL
using namespace std;
typedef long long ll;
typedef pair<int,int> P;

int n;
int a[2005];
int sum[2005];
int cnt[2005];
vector<ll> dp[2005];

int main(void){
	scanf("%d",&n);
	for(int i=0;i<n;i++){
		scanf("%d",&a[i]);
		sum[i+1]+=a[i];
		sum[i+1]+=sum[i];
	}
	dp[0].push_back(1);
	for(int i=0;i<n;i++){
		int vi=0;
		while(vi<=sum[i+1]+100){
			dp[i+1].push_back(0);
			vi+=60;
		}
		int siz=a[i]/60;
		int dif=a[i]%60;
		for(int j=0;j<dp[i].size();j++){
			dp[i+1][j]|=dp[i][j];
			if(dif>0){
				dp[i+1][j+siz+1]|=(dp[i][j]&((1LL<<60)-(1LL<<(60-dif))+1LL))>>(60-dif);
				dp[i+1][j+siz]|=(dp[i][j]&((1LL<<(60-dif))-1LL))<<dif;
			}else{
				dp[i+1][j+siz]|=dp[i][j];
			}
			//cout << ((dp[i][j].to_ullong()&((1LL<<dif)-1))<<dif)  << endl;
		}
		dp[i].clear();
	}
	for(int i=0;i<dp[n].size();i++){
		for(int j=0;j<60;j++){
			if(dp[n][i]>>j & 1){
				if((i*60+j)*2>=sum[n]){
					printf("%d\n",i*60+j);
					return 0;
				}
			}
		}
	}
	return 0;
}
