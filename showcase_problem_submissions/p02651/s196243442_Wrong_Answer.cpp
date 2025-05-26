#include <bits/stdc++.h>
using namespace std;
using lint = long long int;
using P = pair<lint, lint>;
#define FOR(i, begin, end) for(int i=(begin),i##_end_=(end);i<i##_end_;i++)
#define IFOR(i, begin, end) for(int i=(end)-1,i##_begin_=(begin);i>=i##_begin_;i--)
#define REP(i, n) FOR(i,0,n)
#define IREP(i, n) IFOR(i,0,n)
#define ALL(a)  (a).begin(),(a).end()
constexpr int MOD = 1000000007;
constexpr int INF = 2147483647;
void yes(bool expr) {
  cout << (expr ? "Yes" : "No") << "\n";
}
int gauss_jordan(vector<vector<int>> &A) {
    int m = A.size(), n = A[0].size();
    int rank = 0;
    for (int col = 0; col < n; ++col) {
        // ピボットを探す
        int pivot = -1;
        int ma = 0;
        for (int row = rank; row < m; ++row) {
            if (abs(A[row][col]) > ma) {
                ma = abs(A[row][col]);
                pivot = row;
            }
        }
        // ピボットがなかったら次の列へ
        if (pivot == -1) continue;

        // まずは行を swap
        swap(A[pivot], A[rank]);

        // ピボットの値を 1 にする
        //auto fac = A[rank][col];
        //for (int col2 = 0; col2 < n; ++col2) A[rank][col2] /= fac;

        // ピボットのある列の値がすべて 0 になるように掃き出す
        for (int row = 0; row < m; ++row) {
            if (row != rank && abs(A[row][col]) > 0) {
                auto fac = A[row][col];
                for (int col2 = 0; col2 < n; ++col2) {
                    A[row][col2] = A[row][col2] ^ (A[rank][col2] * fac);
                }
            }
        }
        ++rank;
    }
    return rank;
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    REP(t, T) {
        int N;
        cin >> N;
        vector<lint> A(N);
        REP(i, N) cin >> A[i];
        string S;
        cin >> S;
        vector<vector<int>> X;
        vector<vector<int>> Y;
        bool mode = false;
        IREP(i, N) {
            if(S[i] == '1') {
                mode = true;
                vector<int> tmp(60);
                REP(k, 60) if((A[i]>>k&1)==1) tmp[k] = 1;
                X.push_back(tmp);
            } else if(!mode) {
                vector<int> tmp(60);
                REP(k, 60) if((A[i]>>k&1)==1) tmp[k] = 1;
                Y.push_back(tmp);
            }
        }
        if(X.size() == 0) {
            cout << 0 << endl;
            continue;
        } else if(Y.size() == 0) {
            cout << 1 << endl;
            continue;
        }
        gauss_jordan(X);
        gauss_jordan(Y);
        //REP(i, X.size()) {
        //    REP(j, X[0].size()) cout << X[i][j] << " ";
        //    cout << endl;
        //}
        bool flg3 = true;
        REP(i, X.size()) {
            bool flg2 = false;
            REP(j, Y.size()) {
                bool flg = true;
                REP(k, 60) {
                    if(X[i][k] != Y[j][k]) {
                        flg = false;
                        break;
                    }
                }
                if(flg) {
                    flg2 = true;
                    break;
                }
            }
            if(!flg2) {
                flg3 = false;
                break;
            }
        }
        if(!flg3) {
            cout << 1 << "\n";
        } else {
            cout << 0 <<  "\n";
        }

        
        
    }
}