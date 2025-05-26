#include<iostream>
#include<cstdio>
const int N=205,M=62;
long long b[M];
void init() {
    for(int j=0;j<M;j++) b[j]=0;
}
void insert(long long x) {
    for(int j=M-1;~j;--j) if(x>>j&1) {
        if(!b[j]) {
            b[j]=x;
            break;
        }
        else x^=b[j];
    }
}
bool query(long long x) {
    for(int j=M-1;~j;--j) if(x>>j&1) x^=b[j];
    return x==0;
}
int main() {
    int T,n;char s[N];
    long long a[N];
    scanf("%d",&T);
    while(T--) {
        scanf("%d",&n);
        for(int i=0;i<n;i++) scanf("%lld",a+i);
        scanf("%s",s);
        if(s[n-1]=='1') printf("1\n");
        else {
            int ans=0;
            init();
            for(int i=0;i<n;i++) if(s[i]=='0') insert(a[i]);
            for(int i=0;i<n;i++) if(s[i]=='1'&&!query(a[i])) ans=1;
            printf("%d\n",ans); 
        }
    }
    return 0;
}