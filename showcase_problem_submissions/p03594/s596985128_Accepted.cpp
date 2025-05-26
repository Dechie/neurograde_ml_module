#include <bits/stdc++.h>

using namespace std;
typedef long long ll;

#define REP(i,n) for(int i=0;i<n;++i)
#define SORT(name) sort(name.begin(), name.end())
#define ZERO(p) memset(p, 0, sizeof(p))
#define MINUS(p) memset(p, -1, sizeof(p))

#define MOD 1000000007
#define INF 1000000000

int H, W, d;
int main()
{
    cin >> H >> W >> d;

    if(d % 2 != 0) {
        REP(i, H) {
            char tile1, tile2;
            if(i % 2 == 0) {
                tile1 = 'R';
                tile2 = 'G';
            } else {
                tile1 = 'G';
                tile2 = 'R';
            }
            REP(j, W) {
                if(j % 2 == 0) { printf("%c", tile1); }
                else { printf("%c", tile2); }
            }
            printf("\n");
        }
    } else {
        REP(i, H) {
            char up, down, left, right;
            for(int j = 0; j < W; j += d) {
                if(((i + j) / d) % 2 == 0) {
                    up = 'R';
                    down = 'G';
                    left = 'Y';
                    right = 'B';
                } else {
                    up = 'G';
                    down = 'R';
                    left = 'B';
                    right = 'Y';
                }
                int up_n = d - (i % d) * 2;
                if(i % d >= d / 2) { up_n = 0; }
                int down_n = ((i % d) - d / 2) * 2;
                if(i % d <= d / 2) { down_n = 0; }
                int width_n;
                if(i % d <= d / 2) {
                    width_n = i % d;
                } else {
                    width_n = d - (i % d);
                }
                int ind = j;
                for(int k = 0; k < width_n && ind < W; k++, ind++) { printf("%c", left); }
                for(int k = 0; k < up_n && ind < W; k++, ind++) { printf("%c", up); }
                for(int k = 0; k < down_n && ind < W; k++, ind++) { printf("%c",down); }
                for(int k = 0; k < width_n && ind < W; k++, ind++) { printf("%c", right); }
            }
            printf("\n");
        }
    }
    return 0;
}
