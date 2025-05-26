#include <algorithm>
#include <bitset>
#include <complex>
#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <tuple>
#include <utility>
#include <vector>
using namespace std;

using int64 = int64_t;

constexpr int64 MOD = 1000000007;

char rev(char ch) {
    return (ch == 'B') ? 'A' : 'B';
}

int N, K;
string s;

int main() {
    cin >> N >> K >> s;
    if (s[0] == 'A') {
        --K;
        s[0] = 'B';
    }
    if (s[N-1] == 'B') {
        if (K > 0) {
            --K;
            s[N-1] = 'A';
            for (int j = 0; j < N-1; ++j) {
                s[j] = rev(s[j+1]);
            }
        }
        if (K > 0) {
            --K;
            s[0] = 'B';
        }
    }
    if (K > 0) {
        int slide = 0;
        for (int j = 1; K > 0 && j < N; ++j) {
            slide++;
            if (s[j] == 'A') {
                --K;
            } else {
                K -= 2;
            }
        }
        for (int j = 0; j + slide < N-1; ++j) {
            s[j] = slide%2 == 0 ? s[j+slide] : rev(s[j+slide]);
        }
        for (int j = max(0, N-1-slide); j < N-1; ++j) {
            s[j] = (N-1-j)%2 == 0 ? 'A' : 'B';
        }
        s[0] = (K % 2 == 0) ? 'B' : 'A';
    }
    cout << s << endl;
    return 0;
}
