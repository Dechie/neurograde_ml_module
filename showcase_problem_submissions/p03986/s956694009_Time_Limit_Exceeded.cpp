#include<string>
#include<iostream>
#include <cstdio>
#include <algorithm>
using namespace std;
string str,s;
void delet(string &str,const string &s,int n); 
int main()
{  
    cin>>str;
    s="ST";
    int m,flag=0,sum=0,n=2;
    while(flag==0)
    {
        m=str.find(s);//在str中出现子串的位置 
        if(m<0) flag=1;
        else
        {
          str.erase(m,n);         
          sum++;
        }
    } 
    cout<<str.size()<<endl;
    return 0;
}