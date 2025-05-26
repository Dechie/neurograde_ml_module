#include <bits/stdc++.h>

#define fst(t) std::get<0>(t)
#define snd(t) std::get<1>(t)
#define thd(t) std::get<2>(t)
#define unless(p) if(!(p))
#define until(p) while(!(p))

using ll = std::int64_t;
using P = std::tuple<int,int>;

constexpr ll MOD = 1'000'000'007;
int N, M;
int S[2100], T[2100];
ll dp[2100][2100];

int main(){
    std::cin.tie(nullptr);
    std::ios::sync_with_stdio(false);

    std::cin >> N >> M;

    for(int i=0;i<N;++i){
        std::cin >> S[i];
    }

    for(int i=0;i<M;++i){
        std::cin >> T[i];
    }

    for(int i=N-1;i>=0;--i){
        for(int j=M-1;j>=0;--j){
            dp[i][j] = (dp[i+1][j] + dp[i][j+1] + MOD - dp[i+1][j+1]) % MOD;
            
            if(S[i] == T[j]){
                dp[i][j] = (dp[i][j] + dp[i+1][j+1] + 1) % MOD;
            }
        }
    }

    // for(int i=0;i<=N;++i){
    //     for(int j=0;j<=M;++j){
    //         std::cout << dp[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    ll res = (dp[0][0] + 1) % MOD;
    std::cout << res << std::endl;
}
