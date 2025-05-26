#include <vector> 
#include <list> 
#include <map> 
#include <set> 
#include <deque> 
#include <queue> 
#include <stack> 
#include <bitset> 
#include <algorithm> 
#include <functional> 
#include <numeric> 
#include <utility> 
#include <sstream> 
#include <iostream> 
#include <iomanip> 
#include <cstdio> 
#include <cmath> 
#include <cstdlib> 
#include <cctype> 
#include <string> 
#include <cstring> 
#include <ctime> 

using namespace std;

#define _int64 long long

char s[510000];
int a[1000000];
char ans[510000];

int main()
{
  vector<int> a;
  int i,j,n,k,f,jian,h,t,need;
  scanf("%d%d",&n,&k);
  scanf("%s",s);
  s[n]='B';
  for (i=1;i<n;i++)
    s[n+i]='A'+'B'-s[n+i-1];
  ans[n]='\0';
  for (i=0;i<n;i++)
  {
    if (i==0)
    {
      if (s[0]=='A') need=2;
      else need=1;
    }
    else
    {
      if (s[i]==s[i+n-1])
      {
        need=2;
      }
      else need=1;
    }
    if (k>=need) k-=need;
    else
    {
      for (j=0;j<n;j++)
        if ((i==0)||(s[i+n-1]=='A'))
          ans[j]=s[i+j];
        else ans[j]='A'+'B'-s[i+j];
      if (k==1) ans[0]='A'+'B'-ans[0];
      printf("%s\n",ans);
      return 0;
    }
  }
  if ((s[n]==s[n+n-1])&&(k%2!=0))
  {
    for (j=0;j<n;j++)
      ans[j]=s[n+j];
    ans[0]='B';
    printf("%s\n",ans);
  }
  else
  {
    for (j=0;j<n;j++)
      ans[j]=s[n+j];
    printf("%s\n",ans);
  }
  /*
  for (i=0;i<k;i++)
  {
    j=0;f=1;
    while ((j>=0)&&(j<n))
    {
      if (s[j]=='A')
      {
        s[j]='B';f=-f;
        j+=f;
      }
      else
      {
        s[j]='A';
        j+=f;
      }
    }
    printf("%s\n",s);
  }
  */
  return 0;
}