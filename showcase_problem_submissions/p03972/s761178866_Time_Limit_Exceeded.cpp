class in{struct my_iterator{int it;const bool rev;explicit constexpr my_iterator(int it_, bool rev=false):it(it_),rev(rev){}int operator*(){return it;}bool operator!=(my_iterator& r){return it!=r.it;}void operator++(){rev?--it:++it;}};const my_iterator i,n;public:explicit constexpr in(int n):i(0),n(n){}explicit constexpr in(int i,int n):i(i,n<i),n(n){}const my_iterator& begin(){return i;}const my_iterator& end(){return n;}};

#include <bits/stdc++.h>
using namespace std;

using i64 = long long;

int main() {
    cin.tie(0); ios::sync_with_stdio(false);
    i64 w, h;
    cin >> w >> h;
    vector<i64> p(w);
    for(auto& x : p) cin >> x;
    vector<i64> q(h);
    for(auto& x : q) cin >> x;
    i64 pSum = accumulate(p.begin(), p.end(), 0LL);
    i64 qSum = accumulate(q.begin(), q.end(), 0LL);
    i64 ans = pSum + qSum;
    sort(p.begin(), p.end());
    sort(q.begin(), q.end());
    for(i64 x : p) {
        auto it = lower_bound(q.begin(), q.end(), x);
        i64 i = q.end() - it;
        
        ans += x * i + accumulate(q.begin(), it, 0LL);
    }
    /*
    for(i64 x : q) {
        i64 i = lower_bound(p.begin(), p.end(), x) - p.begin();
        i = w - i;
//        cout << x << ' ' << i << endl;
        ans += x * i;
    }
      */
    cout << ans << endl;
    return 0;
}