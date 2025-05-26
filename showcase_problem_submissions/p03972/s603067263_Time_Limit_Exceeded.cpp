#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int INF = 1e9;
int W,H;
int p[100000];
int q[100000];
ll p_sum = 0;
ll q_sum = 0;
void solve(){
    ll res = p_sum+q_sum;
    for(int i=0;i<W;i++){
        for(int j=0;j<H;j++){
            res += min(p[i],q[j]);
        }
    }
    cout << res <<endl;
}

int main() {
    cin >> W >> H;
    for(int i=0;i<W;i++){
        cin >> p[i];
        p_sum += p[i];
    }
    for(int i=0;i<H;i++){
        cin >> q[i];
        q_sum += q[i];
    }
    solve();
    return 0;
}