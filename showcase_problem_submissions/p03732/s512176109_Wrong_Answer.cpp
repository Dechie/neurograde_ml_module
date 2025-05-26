#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

int main()
{
  int N, W;
  cin >> N >> W;
  vector<int> v1, v2, v3, v4;
  int w, v;
  cin >> w >> v;
  int w1 = w;
  v1.push_back(v);
  for(int i = 1; i < N; i++){
    cin >> w >> v;
    if(w == w1){
      v1.push_back(v);
    }else if(w == w1+1){
      v2.push_back(v);
    }else if(w == w1+2){
      v3.push_back(v);
    }else{
      v4.push_back(v);
    }
  }
  sort(begin(v1),end(v1),greater<int>());
  sort(begin(v2),end(v2),greater<int>());
  sort(begin(v3),end(v3),greater<int>());
  sort(begin(v4),end(v4),greater<int>());
  long long ans = 0;
  for(int i = 0; i <= (int)v1.size(); i++){
    for(int j = 0; j <= (int)v2.size(); j++){
      for(int k = 0; k <= (int)v3.size(); k++){
        for(int l = 0; l <= (int)v4.size(); l++){
          if(i*w1+j*(w1+1)+k*(w1+2)+l*(w1+3) > W) continue;
          long long x = accumulate(begin(v1),begin(v1)+i,0)+accumulate(begin(v2),begin(v2)+j,0)+accumulate(begin(v3),begin(v3)+k,0)+accumulate(begin(v4),begin(v4)+l,0);
          ans = max(ans,x);
        }
      }
    }
  }
  cout << ans << endl;
}
