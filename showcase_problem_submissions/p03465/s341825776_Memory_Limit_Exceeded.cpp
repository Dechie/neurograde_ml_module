#include <cstdio>
#include <cstring>
#include <algorithm>
#include <bitset>
#define ele int
using namespace std;
#define maxn 2010
ele n,a[maxn];
bitset<maxn*maxn> f[maxn];
int main(){
	scanf("%d",&n);
	ele s=0;
	for (int i=0; i<n; ++i) scanf("%d",a+i),s+=a[i];
	f[0]=0;
	for (int i=0; i<n; ++i){
		bitset<maxn*maxn> tmp=f[i]<<a[i];
		tmp[a[i]]=1;
		f[i+1]=f[i] | tmp;
	}
	ele ans=(s+1)/2;
	while (!f[n][ans]) ++ans;
	printf("%d\n",ans);
	return 0;
}