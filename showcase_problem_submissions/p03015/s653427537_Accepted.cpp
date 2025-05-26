#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <unordered_map>
#include <vector>
#include <string.h>
#include <set>

using namespace std;

#define COUT(x) cout << #x << " = " << (x) << " (L" << __LINE__ << ")" << endl

#define rep(i, n) for(int i = 0; i < n; ++i)
#define all(s) s.begin(), s.end()

typedef long long ll;
typedef unsigned long long ull;
typedef pair<ll, ll> P;
typedef tuple<ll, ll, ll> triple;
typedef double D;

const ll INF = 1e9;
const ll MOD = 1000000007;  // 1e9 + 7

ll powmod(ll x, ll n) { // like pow(x, n)
  ll r = 1;
  while (n) {
    if (n & 1) {
      r = r * x % MOD;
    }
    x = x * x % MOD;
    n >>= 1;
  }
  return r;
}

ll dp[1000100][2];  // dp[i][s] .. i: position, s: equal or lower

int main(int argc, char** argv) {
  string L;
  cin >> L;
  int N = L.size();
  dp[0][0] = 1;
  rep(i, N) {
    { // a + b == 0
      if (L[i] == '0') {
        dp[i+1][0] = dp[i][0];
        dp[i+1][1] = dp[i][1];
      } else { // L[i] == '1', so ns is 1
        dp[i+1][1] = (dp[i][0] + dp[i][1]) % MOD;
      }
    }
    { // a + b == 1
      if (L[i] == '0') {
        // We can add only when s == 1
        (dp[i+1][1] += (dp[i][1] * 2) % MOD) %= MOD;
      } else { // L[i] == '1'
        (dp[i+1][0] += (dp[i][0] * 2) % MOD) %= MOD;
        (dp[i+1][1] += (dp[i][1] * 2) % MOD) %= MOD;
      }
    }
  }

  cout << (dp[N][0] + dp[N][1]) % MOD << endl;
}
