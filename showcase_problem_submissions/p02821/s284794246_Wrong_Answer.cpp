#include <bits/stdc++.h>
using namespace std;

int main() {
	int n,m;
	cin>>n>>m;
	int a[n];
	for(int i=0;i<n;i++){cin>>a[i];}
	int ans=0;
	sort(a,a+n);
	int k=m;
	while(k>0)
	{
		for(int i=n-1;i>=0;i--)
		{
			if(k==0){break;}
			ans+=(2*a[i]);
			k--;
			for(int j=i-1;j>=0;j--)
			{
				if(k==0){break;}
				ans+= a[i]+a[j];
				k--;
				if(k==0){break;}
				ans+=a[i]+a[j];
				k--;
			}
			if(k==0){break;}
		}
	}
	cout<<ans;
	return 0;
}
