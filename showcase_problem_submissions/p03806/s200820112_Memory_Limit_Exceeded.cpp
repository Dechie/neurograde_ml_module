#include <iostream>
#include <string>
#include <queue>
#include <stack>
#include <algorithm>
#include <list>
#include <vector>
#include <complex>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
#include <bitset>
#include <ctime>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <numeric>
#include <tuple>
#include <sstream>
#include <fstream>

using namespace std;
#define REP(i, n) for (int (i) = 0; (i) < (n); (i)++)
#define FOR(i, a, b) for (int (i) = (a); (i) < (b); (i)++)
#define RREP(i, a) for(int (i) = (a) - 1; (i) >= 0; (i)--)
#define FORR(i, a, b) for(int (i) = (a) - 1; (i) >= (b); (i)--)
#define DEBUG(C) cerr << #C << " = " << C << endl;
using LL = long long;
using VI = vector<int>;
using VVI = vector<VI>;
using VL = vector<LL>;
using VVL = vector<VL>;
using VD = vector<double>;
using VVD = vector<VD>;
using PII = pair<int, int>;
using PDD = pair<double, double>;
using PLL = pair<LL, LL>;
using VPII = vector<PII>;
template<typename T> using VT = vector<T>;
#define ALL(a) begin((a)), end((a))
#define RALL(a) rbegin((a)), rend((a))
#define SORT(a) sort(ALL((a)))
#define RSORT(a) sort(RALL((a)))
#define REVERSE(a) reverse(ALL((a)))
#define MP make_pair
#define FORE(a, b) for (auto &&a : (b))
#define FIND(s, e) ((s).find(e) != (s).end())
#define EB emplace_back
template<typename T> inline bool chmax(T &a, T b){if (a < b){a = b;return true;}return false;}
template<typename T> inline bool chmin(T &a, T b){if (a > b){a = b;return true;}return false;}

const int INF = 1e9;
const int MOD = INF + 7;
const LL LLINF = 1e18;
const long double EPS = 1e-9;

const int MAX = 50;
int N, Ma, Mb;
int a[MAX], b[MAX], c[MAX];
LL dp[MAX][1000][1000];

int main(void) {
    scanf("%d%d%d", &N, &Ma, &Mb);
    REP(i, N) {
        scanf("%d%d%d", a + i, b + i, c + i);
    }
    REP(i, MAX)REP(j, 1000) REP(k, 1000) {
        dp[i][j][k] = LLINF;
    }
    dp[0][0][0] = 0;
    REP(i, N) {
        REP(j, 1000) REP(k, 1000) {
            chmin(dp[i + 1][j][k], dp[i][j][k]);
            if (a[i] + j < 1000 && b[i] + k < 1000) {
                chmin(dp[i + 1][a[i] + j][b[i] + k], dp[i][j][k] + c[i]);
            }
        }
    }
    int ma = Ma, mb = Mb;
    LL ans = LLINF;
    while (ma < 1000 && mb < 1000) {
        chmin(ans, dp[N][ma][mb]);
        ma += Ma; mb += Mb;
    }
    if (ans == LLINF) ans = -1;
    cout << ans << endl;
}
