#include <bits/stdc++.h>
#define _overload3(_1, _2, _3, name, ...) name
#define _rep(i, n) repi(i, 0, n)
#define repi(i, a, b) for (ll i = (ll)(a); i < (ll)(b); ++i)
#define rep(...) _overload3(__VA_ARGS__, repi, _rep, )(__VA_ARGS__)
#define ll long long
#define lld long double
#define ALL(x) x.begin(), x.end()
#ifdef DEBUG
#define line() cerr << "[" << __LINE__ << "] ";
#define dump(i) cerr << #i ": " << i << " ";
#define dumpl(i) cerr << #i ": " << i << endl;
#else
#define line(i)
#define dump(i)
#define dumpl(i)
#endif
#define MOD (ll)(1e9 + 7)

using namespace std;

ll modpow(ll n)
{
    if (n == 0)
        return 1ll;
    ll prev = modpow(n / 2);
    return (prev * prev * (n % 2 == 1 ? 2 : 1)) % MOD;
}

int main()
{

    ll n, m;
    cin >> n >> m;
    ll s[n], t[m];

    rep(i, n) cin >> s[i];
    rep(i, m) cin >> t[i];
    ll ans = 0;
    map<ll, bool> u;
    rep(i, n)
    {
        if (u[i])
            continue;
        ll last = -1;
        ll lp = 0;
        ll mlen = 0;
        map<ll, ll> mp;
        bool f = false;
        rep(j, m)
        {
            if (s[i] == t[j])
            {

                u[i] = true;
                f = true;
                if (s[i] != last)
                {
                    mlen++;
                    last = s[i];
                }
                else
                {
                    mp[last]++;
                }
            }
        }
        ll nowans = modpow(mlen);
        if (i != 0)
            nowans--;
        for (auto itr = mp.begin(); itr != mp.end(); itr++)
        {
            ll key = itr->first;
            ll val = itr->second;
            nowans += (val + 1) * modpow(mlen - 1);
            nowans %= MOD;
        }
        ans += nowans;
        ans %= MOD;
    }

    cout << ans << endl;
    return 0;
}