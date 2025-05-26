#include<algorithm>
#include<cmath>
#include<iomanip>
#include<iostream>
#include<map>
#include<numeric>
#include<queue>
#include<set>
#include<sstream>
#include<unordered_map>
#include<unordered_set>
#include<vector>
using uint = unsigned int;
using ll = long long;
enum : int { M = (int)1e9 + 7 };
enum : ll { MLL = (ll)1e18L + 9 };
using namespace std;
#ifdef LOCAL
#include"rprint2.hpp"
#else
#define FUNC(name) template <ostream& out = cout, class... T> void name(T&&...){ }
FUNC(prints) FUNC(printe) FUNC(printw) FUNC(printew) FUNC(printb) FUNC(printd); FUNC(printde);
#endif

int main(){
    cin.tie(0);
    ios::sync_with_stdio(false);
    int h, w, d;
    cin >> h >> w >> d;
    vector<vector<char>> cs(h, vector<char> (w, 'x'));
    char colors[] = {'R', 'Y', 'G', 'B'};
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            int y = i - j + w, x = i + j;
            cout << colors[y / d % 2 * 2 + x / d % 2];
        }
        cout << '\n';
    }
}
