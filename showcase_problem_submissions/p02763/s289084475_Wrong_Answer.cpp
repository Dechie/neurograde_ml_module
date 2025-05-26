#include <iostream>
#include <set>
#include <string>

using namespace std;

set<int> st[26];
set<int>::iterator it;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    string s;
    cin >> n >> s >> q;
    for (int i = 0; i < n; i++) {
        st[s[i] - 'a'].insert(i + 1);
    }
    int t, x, y, res[q];
    char c;
    for (int q1 = 0; q1 < q; q1++) {
        cin >> t;
        if (t == 1) {
            cin >> x >> c;
            st[s[x - 1] - 'a'].erase(x);
            s[x - 1] = c;
            st[c - 'a'].insert(x);
        } else {
            cin >> x >> y;
            res[q1] = 0;
            for (int i = 0; i < 26; i++) {
                it = lower_bound(st[i].begin(), st[i].end(), x);
                if (it != st[i].end() && *it <= y) res[q1]++;
            }
        }
    }
    for (int i = 0; i < q; i++) cout << res[i] << "\n";
    return 0;
}
