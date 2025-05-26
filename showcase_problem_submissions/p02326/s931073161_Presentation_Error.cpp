#include<algorithm>
#include<cstdio>
#include<string>
#include<cstring>
using namespace std;
const int mmax=1400+50;
int dp[mmax][mmax],dot[mmax][mmax];
int mmin(int a,int b,int c){
    int mid[3];
    mid[0]=a,mid[1]=b,mid[2]=c;
    sort(mid,mid+3);
    return mid[0];
}
int main()
{
    int h,w,m;
    scanf("%d%d",&h,&m);
    for(int i=1;i<=h;i++)
        for(int j=1;j<=m;j++)
        scanf("%d",&dot[i][j]);
    for(int i=1;i<=h;i++){
        for(int j=1;j<=m;j++){
            if(dot[i][j] == 1)
                dp[i][j]=0;
            else{
                dp[i][j] = mmin(dp[i-1][j-1],dp[i][j-1],dp[i-1][j])+1;
                }
        }
    }
    int ans=0;
    for(int i=0;i<=h;i++)
        for(int j=0;j<=m;j++)
        ans=max(ans,dp[i][j]);

    printf("%d",ans*ans);

    return 0;
}

