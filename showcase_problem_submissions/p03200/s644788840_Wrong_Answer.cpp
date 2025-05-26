#include <iostream>
#include <cstdio>

#include <vector>

#include <algorithm>

using namespace std;

typedef long long ll;

const int N = 200010;

ll ans;
int cnt;
char s[N];

int main()
{
	scanf("%s", s);
	for (char* p = s; *p; ++p)
		if (*p == 'B')
			++cnt;
		else
			ans += cnt;
	printf("%d\n", ans);
	return 0;
}
