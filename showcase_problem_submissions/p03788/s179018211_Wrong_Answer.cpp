#define _CRT_SECURE_NO_WARNINGS
#include <vector>
#include <iostream>
#include <string.h>
#include <set>
#include <algorithm>
#include <cmath>
#include <string>
using namespace std;

typedef double LD;
typedef long long LL;
typedef pair<int, int> PII;
#define MP make_pair
#define PB push_back
#define FOR(i,a,b) for(int i = (a); i < (b); ++i)
#define RFOR(i,b,a) for(int i = (b) - 1; i >= (a); --i)
#define REP(i, t) FOR(i,0,t)
#define ALL(a) a.begin(), a.end()
#define SZ(a) (int)((a).size())

const LL MOD = 1000000007;
const int INF = 1e9;
const LL LINF = 1LL * INF * INF;
const int MAXN = 200007;
const LD EPS = 1e-7;

string S;
bool L[MAXN];

void brute(string &T)
{
	int pos = 0, w = 1;
	while (pos >= 0 && pos < SZ(T))
	{
		if (T[pos] == 'A')
		{
			T[pos] = 'B';
			w *= -1;
		}
		else
			T[pos] = 'A';
		pos += w;
	}
}

int main()
{
	ios_base::sync_with_stdio(0);
	//freopen("In.txt", "r", stdin);
	int n, k;
	cin >> n >> k >> S;
	bool o = k & 1;
	k = min(k, 2 * n + 2);
	if ((k ^ o) & 1)
		k ^= 1;
	int it = 0;
	while (k && it++ < 7)
	{
		--k;
		brute(S);
	}
	while (k && (S[0] == 'A' || S[n - 1] == 'B'))
	{
		--k;
		brute(S);
	}
	if (k == 0)
	{
		cout << S;
		return 0;
	}
	string T = S;
	brute(T);
	brute(T);
	if (T == S)
		k &= 1;
	int p = 0, cnt = 0;
	REP(i, n)
	{
		L[i] = (i == n - 1 || S[i + 1] != S[i]);
		cnt += L[i];
	}
	while (p < n && k--)
	{
		if (cnt & 1)
		{
			if (L[p])
				--cnt;
			else
				++cnt;
			L[p] ^= 1;
			continue;
		}
		if (!L[p++])
			++cnt;
	}
	int pr = 0;
	while (p < n)
	{
		cout << ((cnt & 1) ? 'A' : 'B');
		if (L[p++])
			--cnt;
		++pr;
	}
	while (pr < n)
	{
		cout << (((n - pr) & 1) ? 'A' : 'B');
		++pr;
	}
	//cout << endl; system("pause");
	return 0;
}