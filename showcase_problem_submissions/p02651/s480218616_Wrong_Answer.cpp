#include <bits/stdc++.h>
using namespace std;
using ll = long long;
typedef pair<int, int> P;
ll Mod = 1000000007;
int main() {
  int T;
  cin >> T;
  for (int i = 0; i < T; i++) {
    int N;
    cin >> N;
    ll A[N];
    string S;
    bool ans = true;
    for (int i = 0; i < N; i++) {
      cin >> A[i];
    }
    cin >> S;
    for (int bit = 0; bit < 63; bit++) {
      int now = 0;
      for (int i = 0; i < N; i++) {
        if (S[i] == '1') {
          if (now == 0 && (A[i] & (1 << bit))) {
            now = 1;
          }
        }
        if (S[i] == '0') {
          if (now == 1 && (A[i] & (1 << bit))) {
            now = 0;
          }
        } 
      }
      if (now == 1) {
        ans = false;
      }
    }
    if (ans) {
      cout << 0 << endl;
    } else {
      cout << 1 << endl;
    }
  }
  return 0;
}