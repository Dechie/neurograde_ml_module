#include "bits/stdc++.h"
using namespace std;
#define CK(N, A, B) (A <= N && N < B)
#define REP(i, a, b) for (int i = a; i < b; i++)
#define RREP(i, a, b) for (int i = (b - 1); a <= i; i--)
#define F first
#define S second
typedef long long ll;

const int INF = 1e9 + 7;
const long long LLINF = 1e18;

int main() {
    string x;
    cin>>x;

    stack<char> stk;
    REP(i, 0, x.size()){
        if(x[i]=='S') stk.push(x[i]);
        else{
            if(stk.empty()) stk.push(x[i]);
            else if(stk.top()=='T') stk.push(x[i]);
            else stk.pop();
        }
    }
    cout<<stk.size()<<endl;
    return 0;
}