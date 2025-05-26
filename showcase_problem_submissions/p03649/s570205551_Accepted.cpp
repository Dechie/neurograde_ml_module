#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
// #define int long long

#define SZ(x) ((int)(x).size())
#define ALL(x) (x).begin(),(x).end()

#define FOR(i,a,b) for(int i=(int)(a);i<(int)(b);i++)
#define REP(i,x) for(int i=0;i<(int)(x);i++)
#define REPS(i,x) for(int i=1;i<=(int)(x);i++)
#define RREP(i,x) for(int i=((int)(x)-1);i>=0;i--)
#define RREPS(i,x) for(int i=((int)(x));i>0;i--)

typedef pair<int, int> pint;

#define mp make_pair
#define mt make_tuple
#define pb push_back
#define pf push_front

#define MOD (1000 * 1000 * 1000 + 7)

ll count(vector<ll> &a, int N) {
    vector<ll> cnt(N, 0);
    ll total = 0;

    REP(i, N) {
        cnt[i] = (a[i] >= N) ? a[i] / N : 0;
        total += cnt[i];
    }

    REP(i, N) {
        a[i] -= cnt[i] * N;
        a[i] += total - cnt[i];
    }

    return total;
}

bool is_end(const vector<ll> &a, int N) {
    REP(i, N) {
        if (a[i] >= N) {
            return false;
        }
    }
    return true;
}

signed main() {
    int N;
    cin >> N;

    vector<ll> a(N);
    REP(i, N) {
        ll _a;
        cin >> _a;
        a[i] = _a;
    }

    ll ans = 0;
    while (is_end(a, N) == false) {
        ans += count(a, N);
    }

    cout << ans << endl;

    return 0;
}