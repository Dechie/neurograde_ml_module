#include <bits/stdc++.h>

#define rep(i, n) for(int i = 0; i < n; i++)

using namespace std;
using ll = long long;
int mod = 1000000007;

int main(void){
    string L; cin >> L;
    //Lは pow(2, 100001)以下の正整数
    //a+b <= L
    //a+b = a XOR b
    //基本的にa+b >= a XOR bで、等号が成立するのは各ビット両方ともが1ではないとき
    ll digit = L.length();
    //2進数でdigit-1桁以下で表せる数字はa+b=a^bな限り条件を満たす.
    //a, bの各桁目について(1, 0), (0, 1), (0, 0)の3通りのいずれかである
    ll ans = 0;
    //で、digit桁で表される整数をどうするか
    //0が初めに先頭からk+1桁目に現れるとする(L[k] = 1)
    //L[0]からL[k-1]までは全て1。その間に(a, b) = (0, 0)がうんぬん

    ll pw3[digit+1] = {1};
    ll pw2[digit+1] = {1};
    for(int i = 1; i < digit+1; i++){
        pw2[i] = (pw2[i-1]*2) % mod;
        pw3[i] = (pw3[i-1]*3) % mod;
    }

    int cnt1[digit+1] = {}; if(L[0] == '1') cnt1[1] = 1;
    for(int i = 1; i < digit; i++){
        if(L[i] == '1') cnt1[i+1] = cnt1[i]+1;
        else cnt1[i+1] = cnt1[i];
    }
    int idx = 0;
    for(int i = 0; i < digit; i++){
        while(L[i] == '1'){
            ans = (ans + pw2[cnt1[i]]*pw3[digit-1-i]) % mod;
            i++;
        }
    }
    cout << (ans+pw2[cnt1[digit]]) % mod << endl;
    return 0;
}