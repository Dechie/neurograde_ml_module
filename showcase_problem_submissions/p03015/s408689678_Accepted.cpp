#include<bits/stdc++.h>
#define fi first
#define se second
#define pb push_back
#define ll long long
using namespace std;
const int inf=1e9;
const int mod=1e9+7;
const int maxn=1e5+10;
ll dp[maxn];
ll a[maxn],b[maxn],f[maxn];
char s[maxn];
ll ksm(ll a,ll b){
	ll ret=1;
	while(b){
		if(b&1) ret=ret*a%mod;
		a=a*a%mod;
		b>>=1;
	}
	return ret;
}
int main(){
	//ios::sync_with_stdio(false);
	//freopen("in","r",stdin);
	cin>>s+1;
	f[0]=1;
	for(int i=1;i<maxn;i++) f[i]=f[i-1]*i%mod;
	int n=strlen(s+1);
	for(int i=n;i>=1;i--) a[n-i+1]=s[i]-'0';
	ll ans=0;
	for(int i=n;i>=1;i--){
		b[i]=b[i+1]+(a[i]==1);
	}
	for(int i=1;i<=n;i++){
		if(a[i]==1){
			ans+=ksm(2,b[i+1])*ksm(3,i-1)%mod;
			ans%=mod;
		}
	}
	ans=(ans+ksm(2,b[1]))%mod;
	cout<<ans<<endl;
	return 0;
}
