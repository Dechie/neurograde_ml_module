#include <iostream>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <utility>
#include <string>

using namespace std;

int main() {
	string S;
	cin >> S;

	int ans = 0;
	int ws = 0;
	for (int i = 0; i < S.size(); i++) {
		if (S[i] == 'W') {
			ans += i - ws;
			ws++;
		}
	}

	cout << ans << endl;
	return 0;
}
