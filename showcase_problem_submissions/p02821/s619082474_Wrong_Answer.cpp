#include<bits/stdc++.h>
using namespace std; 
typedef long long ll;
typedef long double ld;
#define pb push_back
#define fi first
#define se second
const int N = 5e5+5;
const double PI = acos(-1);
typedef complex<double> base;
vector<base> omega;
long long FFT_N , mod = 1e18;
void init_fft(long long n)
{
  FFT_N  = n;
  omega.resize(n);
  double angle = 2 * PI / n;
  for(int i = 0; i < n; i++)
    omega[i] = base( cos(i * angle), sin(i * angle));
}
void fft (vector<base> & a)
{
  long long n = (long long) a.size();
  if (n == 1)  return;
  long long half = n >> 1;
  vector<base> even (half),  odd (half);
  for (int i=0, j=0; i<n; i+=2, ++j)
    {
      even[j] = a[i];
      odd[j] = a[i+1];
    }
  fft (even), fft (odd);
  for (int i=0, fact = FFT_N/n; i < half; ++i)
    {
      base twiddle =  odd[i] * omega[i * fact] ;
      a[i] =  even[i] + twiddle;
      a[i+half] = even[i] - twiddle;
    }
}
void multiply (const vector<long long> & a, const vector<long long> & b, vector<long long> & res)
{
  vector<base> fa (a.begin(), a.end()),  fb (b.begin(), b.end());
  long long n = 1;
  while (n < 2*max (a.size(), b.size()))  n <<= 1;
  fa.resize (n),  fb.resize (n);
 
  init_fft(n);
  fft (fa),  fft (fb);
  for (size_t i=0; i<n; ++i)
    fa[i] = conj( fa[i] * fb[i]);
  fft (fa);
  res.resize (n);
  for (size_t i=0; i<n; ++i)
    {
      res[i] = (long long) (fa[i].real() / n + 0.5);
    }
}

void solve(){
	int n; 
	cin>>n; 
	ll m;
	cin>>m;
	vector<ll>a,b;
	for(int i=1;i<=1e5;i++){
		a.pb(0);
		b.pb(0);
	}
	for(int i=0;i<n;i++){
		int x; 
		cin>>x;
		a[x]++;
		b[x]++;
	}
	vector<ll>res;
	multiply(a,b,res);
	int sz = res.size();
	ll ans = 0;
	for(ll i=sz-1;i>=0;i--){
		ll d = min(res[i],m);
		m-=d;
		ans+=d*i;
	}
	cout<<ans<<endl;
}
int main(){
	ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0);
	int t=1; 
//	cin>>t; 
	while(t--){
		solve();
	}
	return 0;
}
