#include <iostream>
#include <vector>
#include <memory>
const int MAX_N = 524288;

using namespace std;

int tree[MAX_N*2-1][256];
void query(vector<int>& result, int a, int b, int k, int l, int r) {
    if (r <= a || b <= l) return;
    if (a <= l && r <= b) {
        for (int i=0; i<256; i++) {
            result[i] += tree[k][i];
        }
    }
    else {
        query(result, a, b, k*2+1, l, (l+r)/2);
        query(result, a, b, k*2+2, (l+r)/2, r);
    }
}
int main() {
    int N;
    char S[MAX_N];
    cin >> N;
    cin >> S;

    int Q;
    cin >> Q;
    //cerr << "Q=" << Q << endl;

    for (int i=0; i<N; i++) {
        char s = S[i];
        int node = MAX_N - 1 + i; // leaf
        tree[node][s] += 1;
        while (node > 0) {
            node = (node-1) / 2;
            tree[node][s] += 1;
        }
    }

    for (int i=0; i<Q; i++) {
        int qtype, q0;
        cin >> qtype >> q0;
        if (qtype == 1) {
            char q1; cin >> q1;
            q0--; //0-indexed

            int node = MAX_N - 1 + q0;
            char s = S[q0];
            tree[node][s] -= 1;
            tree[node][q1] += 1;
            while (node > 0) {
                node = (node-1) / 2;
                tree[node][s] -= 1;
                tree[node][q1] += 1;
            }
            //cerr << "update done" << endl;
        }
        if (qtype == 2) {
            int q1; cin >> q1;
            q0--; q1--; // 0-indexed

            vector<int> result(256);
            query(result, q0, q1+1, 0, 0, MAX_N);
            int answer = 0;
            for (int j=0; j<256; j++) {
                if (result[j] > 0) {
                    answer++;
                    //cerr << (char)j;
                }
            }
            //cerr << endl;
            cout << answer << endl;
        }
        //cerr << "query done" << endl;
    }
    return 0;
}

