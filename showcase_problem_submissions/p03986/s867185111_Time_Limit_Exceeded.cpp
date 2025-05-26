#include<iostream>
#include<ctype.h>
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
inline int read(){
	int x=0,f=0;char ch=getchar();
	while(!isdigit(ch))f|=ch=='-',ch=getchar();
	while(isdigit(ch))x=x*10+(ch^48),ch=getchar();
	return f?-x:x;
}
char s[2000007];
int er,ans;
int main(){
	cin>>s;
	for(int i=0;i<=strlen(s);i++){
		if(s[i]=='S')++er;
		if(s[i]=='T' && er>0)--er,ans++;
	}
	cout<<strlen(s)-ans-ans<<"\n";
	return 0;
}