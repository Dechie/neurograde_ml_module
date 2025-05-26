#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <set>
#include <stack>
#include <functional>
#define rep(i, n) for(int i = 0; i < n; i++)
#define ll long long
using namespace std;


int main()
{
	cin.tie(0);
	ios::sync_with_stdio(false);
    int debt = 100000,   n;
    cin >> n;
    rep(i, n) {
        debt *= 1.05;
        if (debt % 1000 > 0) {
            debt = (debt / 1000 + 1) * 1000;
        }
    }
    cout << debt << "\n";
	return 0;
}
