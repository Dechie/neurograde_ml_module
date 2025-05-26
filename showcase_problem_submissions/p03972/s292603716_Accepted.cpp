#include <bits/stdc++.h>
#define rep(i,n) for(int i=(0);i<(n);i++)

using namespace std;

typedef long long ll;
typedef priority_queue<ll, vector<ll>, greater<ll>> PQ;


void connect(ll& ans, PQ& pq, ll& cnta, ll& cntb){
	ans += pq.top() * cntb;
	pq.pop();
	cnta--;
}

int main()
{
	cin.tie(0);
	ios::sync_with_stdio(false);

	int W, H;
	cin >> W >> H;

	PQ pq1;
	rep(i, W) {
		ll p;
		cin >> p;
		pq1.push(p);
	}

	PQ pq2;
	rep(i, H) {
		ll q;
		cin >> q;
		pq2.push(q);
	}

	ll cnt1 = W + 1;
	ll cnt2 = H + 1;

	ll ans = 0;
	while(!pq1.empty() || !pq2.empty()){
		if(pq1.empty()){
			connect(ans, pq2, cnt2, cnt1);
			// ans += pq2.top() * cnt1;
			// pq2.pop();
			// cnt2--;
			continue;
		}

		if(pq2.empty()){
			connect(ans, pq1, cnt1, cnt2);
			continue;
		}

		if(pq1.top() < pq2.top()){
			connect(ans, pq1, cnt1, cnt2);
			continue;
		}

		connect(ans, pq2, cnt2, cnt1);
	}

	cout << ans << endl;
}
