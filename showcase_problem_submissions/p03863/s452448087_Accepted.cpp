#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <utility>
#include <queue>
using namespace std;
//#define local
int main()
{
#ifdef local
    freopen("input.txt","r",stdin);
    //freopen("output.txt","w",stdout);
#endif
    char s[100010];
    cin>>s;
    int k=strlen(s);
    if(s[0]==s[k-1]&&k%2||s[0]!=s[k-1]&&k%2==0) cout<<"Second";
    else cout<<"First";
    return 0;
}
