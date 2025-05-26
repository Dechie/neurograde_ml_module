#include <bits/stdc++.h>
using namespace std;

int main() {
    cin.tie(0);
    ios::sync_with_stdio(false);
    string s;
    cin >> s;
    int ans = 0;
    int w = 0;
    bool flag = false;
    for (int i = 0; i < s.size(); i++) {
        if (s[i] == 'W') {
            w++;
            if (flag) {
                ans += i - w + 1;
            }
        } else {
            flag = true;
        }
    }
    cout << ans;
}