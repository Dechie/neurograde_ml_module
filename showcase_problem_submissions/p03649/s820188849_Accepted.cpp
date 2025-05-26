#include <iostream>
#include <iomanip>
#include <ios> 
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <queue>
#include <stack>
#include <set>
#include <cmath>
#include <bitset>
#include <map>
#define rep(i,n) for(int (i)=0;(i)<(n);(i)++)
#define rep1(i,n) for(int i=1;i<=(int)(n);i++)
#define sz(c) ((int)(c).size())
#define mp make_pair

using namespace std;
typedef long long ll;
typedef vector<int> vi;
typedef vector<ll> vll;
//typedef pair<int, int> P;
//typedef pair<int, pair<int, int> > PP;
//const int MAX=;



int main(){
	int N;
	cin>>N;
	ll a[50];
	rep(i,N)cin>>a[i];
	
	ll ans=0;
	while(true){
		ll b[50];
		ll s=0;
		rep(i,N){
			b[i]=a[i]/(ll)N;
			s+=b[i];
		}
		ans+=s;
		rep(i,N) a[i]=a[i]%(ll)N+s-b[i];
		
		sort(a, a+N);
		if(a[N-1]<N)break;
/*
		else{
			ans+=a[0]*(ll)N;
			rep(i,N)a[i]-=a[0];
		}
*/
	}
	cout<<ans<<endl;
}