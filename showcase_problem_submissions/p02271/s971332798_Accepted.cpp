#include <bits/stdc++.h>
using namespace std;

#define REP(i, start, count) for(int i=(start); i<(int)(count); ++i)
#define rep(i, count) REP(i, 0, count)
#define ALLOF(c) (c).begin(), (c).end()

typedef long long ll;
typedef unsigned long long ull;

int n, q;
vector<int> A;

bool solve(int i, int m) {
    if ( m == 0 ) return true;
    if ( i >= n ) return false;
    // 左はi番目の要素を選ばない場合
    // 右はi番目の要素を選んだ場合
    return solve(i + 1, m) || solve(i + 1, m - A[i]);
}

// bool solve(int m) {
//     for (int bit = 0; bit < (1 << n); ++bit) {

//         int sum = 0;
//         for (int i = 0; i < n; ++i) {
//             if (bit & (1 << i)) {
//                 sum += A[i];
//                 if (sum > m) break;
//             }
//         }

//         if (sum == m) return true;
//     }
//     return false;
// }

int main(void) {

    cin >> n;
    rep(i, n) {
        int a;
        cin >> a;
        A.push_back(a);
    }

    cin >> q;
    rep(i, q) {
        int m;
        cin >> m;
        if (solve(0, m)) {
            cout << "yes" << endl;
        }
        else {
            cout << "no" << endl;
        }
    }
    return 0;
}
