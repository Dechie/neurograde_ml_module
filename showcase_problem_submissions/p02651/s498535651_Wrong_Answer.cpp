#include<bits/stdc++.h>
using namespace std;
int n,TEST,a[220],pos[2],sop[2],X,F,ff,fla; 
string s;
long long aa[220];
void solve(){
	cin>>n;
	for(int i=1;i<=n;i++) cin>>aa[i];
	cin>>s;s="?"+s;
	F=1;
	for(int i=1;i<=63;i++){
		for(int j=1;j<=n;j++) a[j]=aa[j]%2LL,aa[j]/=2LL;
		fla=0;
		for(int j=1;j<=n;j++) if(aa[j]) fla=1;
		if(!fla) break;
		ff=0;
		for(int j=1;j<=n;j++){
			if(s[j]=='1') continue;
			if(s[j-1]=='0') continue;
			X=0;
			for(int k=j;k<=n;k++){
				X^=a[k];
				if(X==1){ff=1;break;}
			}
			if(ff) break;
		}
		F&=ff;
	}
	cout<<!F<<endl;
}
int main(){
	#ifndef DavidLaptop
		#ifdef FILIN
			freopen(FILIN,"r",stdin);
			freopen(FILOUT,"w",stdout);
		#endif
	#endif
	ios::sync_with_stdio(false);
	cin.tie(NULL);
	cout.tie(NULL);
	cin>>TEST;
	while(TEST--) solve();
	return 0;
}

