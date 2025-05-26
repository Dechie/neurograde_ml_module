#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <iostream>
#include <sstream>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <utility>
#include <numeric>
#include <algorithm>
#include <bitset>
#include <complex>
#include <array>
#include <list>
#include <stack>
#include <valarray>

using namespace std;

typedef unsigned uint;
typedef long long Int;
typedef unsigned long long UInt;

const int INF = 1001001001;
const Int INFLL = 1001001001001001001LL;

template<typename T> void pv(T a, T b) { for (T i = a; i != b; ++i) cout << *i << " "; cout << endl; }
template<typename T> void chmin(T& a, T b) { if (a > b) a = b; }
template<typename T> void chmax(T& a, T b) { if (a < b) a = b; }
int in() { int x; scanf("%d", &x); return x; }
double fin() { double x; scanf("%lf", &x); return x; }
Int lin() { Int x; scanf("%lld", &x); return x; }

int main() {
  int N = in();
  Int A[64], S = 0;
  for (int i = 0; i < N; ++i) {
    A[i] = lin();
    S += A[i];
  }
  sort(A, A + N);
  reverse(A, A + N);

  if (A[0] < N) {
    puts("0");
    return 0;
  }

  Int T[64], s;
  Int K = S - N * (N - 1) - 1;

  Int lo = 0, hi = K + 1;
  while (hi - lo > 1) {
    Int mid = (hi + lo) / 2;
    Int tot = mid, C[64];
    C[0] = mid;
    for (int i = 1; i < N; ++i) {
      C[i] = mid - (A[0] - A[i]) / (N + 1) - 1;
      tot += max(C[i], 0LL);
    }
    if (tot < K) {
      lo = mid;
      s = tot;
      for (int i = 0; i < N; ++i) {
        T[i] = A[i] - C[i] * N + (tot - C[i]);
      }
    } else {
      hi = mid;
    }
  }

  while (*max_element(T, T + N) >= N) {
    (*max_element(T, T + N)) -= N + 1;
    for (int i = 0; i < N; ++i) {
      ++T[i];
    }
    ++s;
  }

  printf("%lld\n", s);

  return 0;
}
