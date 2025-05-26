#include <bits/stdc++.h>
#include <iostream>
using namespace std;


int* mallocint(int si)
{
    return (int*) malloc(si * sizeof(int));
}

int* inputint(int n)
{
    int *ans = mallocint(n);
    for( int i = 0; i < n; i++ ) cin >> ans[i];
    return ans;
}

//　合計がx以上になる探し方がどれくらいあるか
int bsearchcount(int x, int n, int *a)
{
    int pindex = n - 1;
    int ans = 0;
//    cout << "x = " << x ;
    for( int i = 0; i < n; i++ )
    {
        while( pindex >= 0 && a[pindex] + a[i] < x ) 
        {
            pindex--;
        }
        ans += pindex + 1;
    }
//    cout << "  ans = " << ans << endl;
    return ans;
}

int bsearch(pair<int,int> window, int m, int n, int *a)
{
//    cout << "<" << window.first << "," << window.second << ">" << endl;
    if ( window.first  + 1 == window.second ) return window.first;
    int border = window.first + (window.second - window.first) / 2;
    int bcount = bsearchcount(border, n, a);
    if( bcount >= m )
    {
        if( border + 1 == window.second )
        {
            return border;
        }
        return bsearch( make_pair(border, window.second), m, n, a );
    }
    else
    {
        if( window.first + 1 == border ) return window.first;
        return bsearch( make_pair(window.first, border), m, n, a );
    }
}

int getans(int x, int n, int *a, int m)
{
    long long pindex = n - 1;
    long long ans = 0;
    long long psum = 0;
    long long pcount = 0;
    int minsum = a[0] + a[0];
    for( int i = 0; i < n; i++ ) psum += a[i];

    for( int i = 0; i < n; i++ )
    {
        while( pindex >= 0 && a[pindex] + a[i] < x ) 
        {
            psum -= a[pindex];
            pindex--;
        }
//        cout << " pindex = " << pindex << ", psum = " << psum << endl; 
        ans += psum + a[i] * (pindex + 1);
        if( pindex >= 0 && a[i] + a[pindex] < minsum ) minsum = a[i] + a[pindex];
        pcount += pindex + 1;
    }
    ans -= (pcount - m) * minsum;
    return ans;
 
}

int main()
{
    int n, m;
    cin >> n >> m;
    int *a = inputint(n);

    sort(a + 0, a + n, greater<>());

    int borderx = bsearch(make_pair(a[n-1] + a[n-1], a[0] + a[0] + 1), m, n, a);
//    cout << borderx << endl;
    cout << getans(borderx, n, a, m);
    return 0;
}
