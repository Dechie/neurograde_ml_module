#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <tuple>
#include <vector>

using namespace std;

#define rep(i, n) for (int64_t i = 0; i < (n); i++)
#define irep(i, n) for (int64_t i = 0; i <= (n); i++)
#define rrep(i, n) for (int64_t i = (n)-1; i >= 0; i--)
#define rirep(i, n) for (int64_t i = n; i >= 0; i--)

int main()
{
    int n, m;
    cin >> n >> m;

    vector<int> a(n);
    rep(i, n)
    {
        cin >> a[i];
    }

    sort(a.begin(), a.end());
    vector<int64_t> asum(n + 1);
    asum[n] = 0;
    rrep(i, n)
    {
        asum[i] = asum[i + 1] + a[i];
    }

    int lb = 0, ub = 2 * a[n - 1] + 1;
    int64_t lbM = (int64_t)n * n;
    int64_t lbScore = asum[0] * 2 * n;

    while (ub - lb > 1)
    {
        int mid = (lb + ub) / 2;
        int64_t cnt = 0;
        int64_t score = 0;
        int j = n - 1;
        rep(i, n)
        {
            while (j >= 0 && a[i] + a[j] >= mid)
                j--;
            cnt += n - (j + 1);
            score += asum[j + 1] + (int64_t)(n - (j + 1)) * a[i];
        }

        if (cnt >= m)
        {
            lb = mid;
            lbM = cnt;
            lbScore = score;
        }
        else
        {
            ub = mid;
        }
    }
    //cout << lb << " " << lbM << " " << lbScore << endl;
    cout << lbScore - (lbM - m) * lb << endl;

    return 0;
}