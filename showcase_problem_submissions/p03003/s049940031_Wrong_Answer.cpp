#include <bits/stdc++.h>
using namespace std;
using Int = long long;
constexpr Int MOD = 1e9 + 7;
Int B[2001][2001];
int main()
{
    int N, M; cin >> N >> M;
    vector<int> S(N), T(M);
    for (auto &s : S) cin >> s;
    for (auto &t : T) cin >> t;
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= M; j++) {
            B[i][j] = (
                B[i][j-1] + B[i-1][j] - B[i-1][j-1] +
                (S[i-1] == T[j-1] ? B[i-1][j-1] + 1 : 0)
            ) % MOD;
        }
    }
    cout << (B[N][M] + 1) % MOD << '\n';
    return 0;
}
