#include<bits/stdc++.h>
using namespace std;

#define mygc(c) (c)=getchar_unlocked()
#define mypc(c) putchar_unlocked(c)
void reader(int *x){int k,m=0;*x=0;for(;;){mygc(k);if(k=='-'){m=1;break;}if('0'<=k&&k<='9'){*x=k-'0';break;}}for(;;){mygc(k);if(k<'0'||k>'9')break;*x=(*x)*10+k-'0';}if(m)(*x)=-(*x);}
template <class T, class S> void reader(T *x, S *y){reader(x);reader(y);}
void writer(int x, char c){int s=0,m=0;char f[10];if(x<0)m=1,x=-x;while(x)f[s++]=x%10,x/=10;if(!s)f[s++]=0;if(m)mypc('-');while(s--)mypc(f[s]+'0');mypc(c);}
template<class T> void writerLn(T x){writer(x,'\n');}


#define incID(i, l, r) for(int i = (l); i < (r); i++)
#define inc(i, n) incID(i, 0, n)
template<typename T> bool setmin(T &a, T b) { if(a <= b) { return false; } else { a = b; return true; } }
template<typename T> bool setmax(T &a, T b) { if(b <= a) { return false; } else { a = b; return true; } }


// ---- ----

int n, k;
vector<int> vec[100000];
bool has[100000];
int mi[100000];
int ma[100000];

void prod(int x, int y) {
	if(has[y]) {
		bool flag = true;
		if(has[x] && (mi[x] - mi[y]) % 2 == 0) { flag = false; }
		setmax(mi[x], mi[y] - 1);
		setmin(ma[x], ma[y] + 1);
		has[x] = true;
		if(mi[x] > ma[x]) { flag = false; }
		if(! flag) {
			puts("No");
			exit(0);
		}
	}
}

bool dfs(int node, int pre) {
	if(pre != -1) { prod(node, pre); }
	for(auto && i : vec[node]) {
		if(i != pre) {
			dfs(i, node);
			prod(node, i);
		}
	}
}

int main() {
	reader(&n);
	inc(i, n - 1) {
		int a, b;
		reader(&a, &b);
		a--; b--;
		vec[a].push_back(b);
		vec[b].push_back(a);
	}
	fill(mi, mi + n, -1000000);
	fill(ma, ma + n,  2000000);
	reader(&k);
	inc(j, k) {
		int v, p;
		reader(&v, &p);
		v--;
		mi[v] = p;
		ma[v] = p;
		has[v] = true;
	}
	
	dfs(0, -1);
	dfs(0, -1);
	
	puts("Yes");
	inc(i, n) { writerLn(mi[i]); }
	
	return 0;
}
