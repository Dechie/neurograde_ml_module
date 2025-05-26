#include <algorithm>
#include <cstdio>
#include <iostream>
#include <map>
#include <math.h>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <string.h>
#include <vector>
using namespace std;

#define ll long long
#define INF (1 << 30)
#define INFLL (1LL << 60)

#define FOR(i,a,b) for(ll i = (a);i<(b);i++)
#define REP(i,a) FOR(i,0,(a))
#define MP make_pair

ll n, maxW, w[110], v[110];

ll sortW[4][110];
int how[4] = {};

ll memo[110][110][110][110];
// ll saiki2(int now, int nowIndex, ll nowW){
// 	if(nowIndex == 4) return 0;
// 	if(memo[now][nowIndex] != -1) return memo[now][nowIndex];
// 	ll a = 0, b = 0;
// 	if(now < how[nowIndex] && nowW + w[0] + nowIndex <= maxW){
// 		a = saiki(now + 1, nowIndex, nowW + w[0] + nowIndex) + -sortW[nowIndex][now];
// 	}
// 	b = saiki(0, nowIndex + 1, nowW);
// 	return memo[now][nowIndex] = max(a, b);
// }

ll saiki(int use0, int use1, int use2, int use3){
	if(use0 > how[0] || use1 > how[1] || use2 > how[2] || use3 > how[3]) return -INF;
	if(use0 * w[0] + use1 * (w[0] + 1) + use2 * (w[0] + 2) + use3 * (w[0] + 3) > maxW) return -INF;

	if(memo[use0][use1][use2][use3] != -1) return memo[use0][use1][use2][use3];

	ll a = 0, b = 0, c = 0, d = 0;

	if(use0 < how[0]) a = saiki(use0 + 1, use1, use2, use3) + -sortW[0][use0];
	if(use1 < how[1]) b = saiki(use0, use1 + 1, use2, use3) + -sortW[1][use1];
	if(use2 < how[2]) c = saiki(use0, use1, use2 + 1, use3) + -sortW[2][use2];
	if(use3 < how[3]) d = saiki(use0, use1, use2, use3 + 1) + -sortW[3][use3];

	return memo[use0][use1][use2][use3] = max(a, max(b, max(c, d)));
}

int main() {
	for(int i = 0;i < 110;i++){
		for(int j = 0;j < 110;j++){
			for(int k = 0;k < 110;k++){
				for(int l = 0;l < 110;l++){
					memo[i][j][k][l] = -1;
				}
			}
		}
	}

	cin >> n >> maxW;
	for(int i = 0;i < n;i++){
		cin >> w[i] >> v[i];
	}
	for(int i = 0;i < n;i++){
		int now = w[i] - w[0];
		sortW[now][how[now]] = -v[i];
		how[now]++;
	}
	for(int i = 0;i < 4;i++){
		sort(sortW[i], sortW[i] + how[i]);
		// for(int j = 0;j < how[i];j++){
		// 	cout << sortW[i][j] << ":";
		// }
		// cout << endl;
	}

	// cout << sortW[0][0] << "," << sortW[0][1];

	cout << saiki(0, 0, 0, 0) << endl;

	return 0;
}