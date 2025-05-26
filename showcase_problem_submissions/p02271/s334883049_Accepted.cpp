#include <iostream>
using namespace std;
int main(){
        int N,a[30];
        cin >> N;
        bool dp[21][2001]={0};
        for(int i=0 ; i<N ; i++ ){
                cin >> a[i];
                dp[i][0]=1;
        }
        dp[0][0]=1;
        for(int i=1 ; i<=N ; i++ ){
                for(int j=1 ; j<=2000 ; j++ ){
                        if(j>=a[i-1])
                                dp[i][j]=dp[i-1][j-a[i-1]]|dp[i-1][j];
                        else
                                dp[i][j]=dp[i-1][j];
                }
        }
        int m;
        cin >> m;
        int x;
        for(int i=0 ; i<m ; i++ ){
                cin >> x;
                if(dp[N][x])cout<<"yes"<<endl;
                else    cout<<"no"<<endl;
        }
        return 0;
}
