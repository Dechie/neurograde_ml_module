#include <iomanip>
#include <iostream>
#include <vector>
#define rep(i, n) for (int i = 0; i < (n); i++)
using namespace std;
const int64_t mod = 1e9 + 7;

int main(void) {
  ios::sync_with_stdio(false);
  string L;
  cin >> L;
  int N = L.size();

  vector<int64_t> DP1(N + 1, 0);  // == L
  vector<int64_t> DP2(N + 1, 0);  //  < L
  DP1[0] = 1;
  rep(i, N) {
    if (L[i] == '0') {
      DP1[i + 1] = DP1[i];
      DP2[i + 1] = (DP2[i] * 3) % mod;
    } else {
      DP1[i + 1] = (DP1[i] * 2) % mod;
      DP2[i + 1] = (DP1[i] * 1 + (DP2[i] * 3 % mod)) % mod;
    }
  }
  int64_t answer = (DP1[N] + DP2[N]) % mod;
  cout << answer << endl;
  return 0;
}
