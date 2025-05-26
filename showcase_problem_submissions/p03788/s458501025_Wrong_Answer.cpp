#include<bits/stdc++.h>
using namespace std;
using Int = long long;
//INSERT ABOVE HERE
signed main(){
  Int n,k;
  string s;
  cin>>n>>k>>s;
  using P = pair<Int, Int>;
  deque<P> v;
  for(Int i=0;i<n;i++){
    if(v.empty()||v.back().first!=s[i]-'A')
      v.emplace_back(s[i]-'A',0);
    v.back().second++;
  }
  Int cnt=0,lst=0;
  while(!v.empty()&&v.back()==P(lst,1)){
    cnt++;
    lst^=1;
    v.pop_back();
  }

  Int t=0;
  auto print=[&](){
    //cout<<t<<" "<<v.front().first<<":";
    for(Int i=0;i<(Int)v.size();i++)
      cout<<string(v[i].second,(t^v[i].first)+'A');
    Int x=lst;
    for(Int i=0;i<cnt;i++){
      x^=1;
      cout<<char('A'+x);
    }
    cout<<endl;
  };
  
  while(k){
    if(v.empty()) break;
    k--;
    //print();
    //for(P p:v) cout<<(p.first^t)<<" "<<p.second<<endl;
    //cout<<endl;
    if(v.front().first^t){
      if(!--v.front().second){
	v.pop_front();
      }
      //cout<<"UKUNICHIA:";print();
      if(!cnt &&(v.back().first^t)){
	//v.back().first^=1;
	v.back().second++;
      }else{
	cnt++;
	lst^=1;
      }
      t^=1;
    }else{
      if(!--v.front().second){
	v.pop_front();
	v.front().second++;
      }else v.emplace_front(t^1,1);      
    }
    //print();
    //for(P p:v) cout<<(p.first^t)<<" "<<p.second<<endl;
    //cout<<endl;
  }
  if((~k&1)&&(n&1)){
    cout<<"B";
    cnt--;
  }
  print();
  return 0;
}
