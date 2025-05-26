#include <iostream>
#include <cstdio>
#include<list>
#include<iomanip>
#include<cmath>
#include<queue>
#include <functional>
#include<stdio.h>
#include<assert.h>
#include<stack>
#include<sstream>
#include <cstdlib>
#include<map>
#include<algorithm>
#include<iostream>
#include<set>
#include<utility>
#include<memory.h>
#include<complex>
#include<string>
#include<vector>
#include<numeric>
using namespace std;
#define ll  long long
#define pb push_back
#define sz(v)               ((int)((v).size()))
#define all(x)          (x).begin(),(x).end()
#define REP(i, v)       for(int i=0;i<sz(v);i++)
using namespace std;
typedef pair<int,int> pii;
ll n,m;
vector<ll> v;
bool ok()
{
	//REP(i,v) cerr << v[i] << " ";
//	cerr << "\n";
	REP(i,v) if(v[i] >= n) return 0;
	return 1;
}
int main()
{
	scanf("%lld",&n);
	ll x;
	for(int i = 0 ; i < n ; ++i)
	{

		scanf("%lld",&x);
		v.pb(x);
	}
	sort(v.rbegin(),v.rend());
	ll ops = 0;
	while(!ok())
	{

		REP(i,v)
	    {
			REP(j,v) if(i != j) v[j] += 1LL*(v[i]/n);
			ops += (1LL*v[i]/n),v[i] = v[i] % n;
		}
		sort(v.rbegin(),v.rend());
	}
	printf("%lld\n",ops);
}
