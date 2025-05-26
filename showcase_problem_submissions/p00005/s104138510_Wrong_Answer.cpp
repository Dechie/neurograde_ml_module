#include <iostream>
#include<string>
#include<cmath>
#include<algorithm>
#include<cctype>
#include<queue>
#include<stack>
#include<stdio.h>
#include<vector>
#include<set>
#include<map>
#include<cstring>
#include<iomanip>
#define rep(i,n) for(int i=0;i<n;i++)
typedef int long long ll;

using namespace std;
int main(){
    ll a,b;
    while(cin>>a){
        cin>>b;
        if(b>a)swap(b,a);
        ll aa=a,bb=b;
        ll o,p,q;
        while(bb!=0){
            o=bb;
            p=aa%bb;
            aa=bb;bb=p;
        }
        ll i=1;
        while(true){
            i++;
            ll ao=a*i;
            if(ao%b==0){
                q=ao;
                break;
            }
        }
        cout<<o<<" "<<q<<endl;
    }
    
    
    
    
    return 0;
}
