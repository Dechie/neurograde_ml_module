#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <utility>

int main() {
  int n;
  std::cin >> n;
  for (int i = 1; true; ++i) {
    if (i * (i + 1) / 2 >= n) {
      for (int j = i; j > 0; --j) {
        if (j <= n) {
          std::cout << j << std::endl;
          n -= j;
        }
      }
      return 0;
    }
  }
}