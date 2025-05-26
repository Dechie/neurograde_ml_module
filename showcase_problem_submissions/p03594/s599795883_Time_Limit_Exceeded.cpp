#include<bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef pair<int, int> PII;
typedef pair<int, PII> PIP;

const int MM = 1e9 + 7;
const double eps = 1e-8;
const int MAXN = 1000 + 10;

int n, m;

void prework(){

}

void read(){

}

char CH[5] = {'R', 'Y', 'G', 'B', 'R'};
int f[MAXN][MAXN];
int g[MAXN][MAXN][5];

const ll P2 = 312394189;
const ll P3 = 468591283;

struct piphash{
	size_t operator()(const PIP &x) const {
		return (x.first + (x.second.first * P2 + x.second.second) % MM * P3) % MM;
	}
};

unordered_set<PIP, piphash> st;
int vis[MAXN][MAXN];

int ADD(int x, int y, int col, int cnt){
	st.erase(PIP(g[x][y][4], PII(x, y)));
	if (g[x][y][col] == 0){
		g[x][y][4]--;
	}
	g[x][y][col] += cnt;
	if (g[x][y][col] == 0){
		g[x][y][4]++;
	}
	st.insert(PIP(g[x][y][4], PII(x, y)));
	if (g[x][y][4] == 0) return 0;
	else return 1;
}

int d;

int dfs(int dep){
	if (dep == n * m + 1){
		for(int i = 1; i <= n; i++){
			for (int j = 1; j <= m; j++)
				putchar(CH[f[i][j]]);
			puts("");
		}
		return 1;
	}
	PIP p = *(st.begin());
	st.erase(st.begin());
	int x = p.second.first;
	int y = p.second.second;
	//cout << dep << ' ' << x << ' ' << y << endl;
	vis[x][y] = 1;
	for (int i = 0; i < 4; i++)
		if ((g[x][y][i] == 0)){
			f[x][y] = i;
			int flag = 1;
			for (int dx = -d; dx <= d; dx++){
				int dy = d - abs(dx);
				if (1 <= x + dx && x + dx <= n && y + dy >= 1 && y + dy <= m && !vis[x + dx][y + dy]){
					flag &= ADD(x + dx, y + dy, i, 1);
				}
				if (dy != 0) {
					dy = -dy;
					if (1 <= x + dx && x + dx <= n && y + dy >= 1 && y + dy <= m && !vis[x + dx][y + dy]){
						flag &= ADD(x + dx, y + dy, i, 1);
					}
				}
			}
			if (flag) if (dfs(dep + 1)) return 1;
			for (int dx = -d; dx <= d; dx++){
				int dy = d - abs(dx);
				if (1 <= x + dx && x + dx <= n && y + dy >= 1 && y + dy <= m && !vis[x + dx][y + dy]){
					ADD(x + dx, y + dy, i, -1);
				}
				if (dy != 0) {
					dy = -dy;
					if (1 <= x + dx && x + dx <= n && y + dy >= 1 && y + dy <= m && !vis[x + dx][y + dy]){
						ADD(x + dx, y + dy, i, -1);
					}
				}
			}
		}
	vis[x][y] = 0;
	st.insert(p);
	return 0;
}

int px[8] = {-2, -1, -1, 0, 0, 1, 1, 2};
int py[8] = {0, -1, 1, -2, 2, -1, 1, 0};

int p[5000][5000];

void solve(int casi){
//	cout << "Case #" << casi << ": ";
	scanf("%d%d%d", &n, &m, &d);
	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= m; j++){
			st.insert(PIP(4, PII(i, j)));
			g[i][j][4] = 4;
			vis[i][j] = 0;
		}
	if (d <= 5000) dfs(1);
	else{
		if (d & 1){
			for (int i = 1; i <= n; i++){
				for (int j = 1; j <= m; j++)
					putchar((i + j) & 1 ? 'R' : 'B');
				puts("");
			}
		}
		else{
//       13 14
//    22 23 24 25
// 31 32 33 34 35 36
//    42 43	44 45
//       53 54
// -2    3 4
// -1    2 5
			queue<PII> Q;
			int dd = d / 2;
			int bias = d;
			for (int i = -dd + 1; i <= dd - 1; i++){
				for (int j = 1 + (abs(i)); j <= d - abs(i); j++){
					p[bias + i][bias + j] = 1;
					Q.push(PII(bias + i, bias + j));
				}
			}
			for (int i = -dd + 1; i <= dd - 1; i++){
				for (int j = 1 + (abs(i)); j <= d - abs(i); j++){
					p[bias + i][bias + j + d] = 2;
					Q.push(PII(bias + i, bias + j + d));
				}
			}
			for (int i = -dd + 1; i <= dd - 1; i++){
				for (int j = 1 + (abs(i)); j <= d - abs(i); j++){
					p[bias + i - dd][bias + j + dd] = 3;
					Q.push(PII(bias + i - dd, bias + j + dd));
				}
			}
			for (int i = -dd + 1; i <= dd - 1; i++){
				for (int j = 1 + (abs(i)); j <= d - abs(i); j++){
					p[bias + i + dd][bias + j + dd] = 4;
					Q.push(PII(bias + i + dd, bias + j + dd));
				}
			}
			while(!Q.empty()){
				int x = Q.front().first, y = Q.front().second;
				Q.pop();
				for (int k = 0; k < 8; k++){
					int xx = x + px[k] * d;
					int yy = y + py[k] * d;
					if (1 <= xx && xx <= n && 1 <= yy && yy <= m && p[xx][yy] == 0){
						p[xx][yy] = p[x][y];
						Q.push(PII(xx, yy));
					}
				}
			}
			for (int i = 1; i <= n; i++){
				for (int j = 1; j <= m; j++)
					putchar(CH[p[i][j]]);
				puts("");
			}
		}
	}
}

void printans(){

}


int main(){
//	std::ios::sync_with_stdio(false);
	prework();
	int T = 1;
//	cin>>T;
	for(int i = 1; i <= T; i++){
		read();
		solve(i);
		printans();
	}
	return 0;
}