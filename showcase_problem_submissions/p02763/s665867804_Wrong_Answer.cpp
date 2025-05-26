#include <bits/stdc++.h>
#define WHOLE(v) (v).begin(), (v).end()
#define REV_WHOLE(v) (v).rbegin(), (v).rend()
using i64 = int64_t;
using namespace std;
template<class F>auto recursive(F f){return[f](auto...a){return f(f,a...);};}
template<class I,class V=typename I::value_type>V sum_up(const I&l,const I&r){V v;for(I i=l;i!=r;i++)v=i==l?*i:v+*i;return v;}
template<class I,class T=iterator_traits<I>>I operator+(I it,int n){for(int i=0;i<n;i++)it++;return it;}
template<class I,class T=iterator_traits<I>>I operator-(I it,int n){for(int i=0;i<n;i++)it--;return it;}
template<class T>using rev_priority_queue=priority_queue<T,vector<T>,greater<T>>;
template<class T>using vector2d=vector<vector<T>>;
struct fixprec{int p;fixprec(int p):p(p){}};
ostream&operator<<(ostream&o,fixprec f){return o<<fixed<<setprecision(f.p);}
void R_YESNO(bool p){cout<<(p?"YES":"NO")<<endl;}
void R_YesNo(bool p){cout<<(p?"Yes":"No")<<endl;}

/*!* [fenwick] *!*/
/*+* フェニック木
     長さNの配列の部分和と更新をO(log N)で行う *+*/
template <typename T, typename F = function<T(T, T)>>
struct Fenwick {
    int N;
    vector<T> array;
    const F f;
    const T id;
    Fenwick() {}
    Fenwick(int len, const F f, const T& id) : N(len), f(f), id(id) {
        array.assign(N, id);
    }
    void update(int k, T val) {
        for (int i = 0, j = k; j < N; i++) {
            array[j] = f(array[j], val);
            if ( (j & (1 << i)) == 0) j |= 1 << i;
        }
    }
    
    // sum of the elements [0:i]
    T sum(int i) {
        i++;
        T s = id;
        while(i) {
            s = f(s, array[i - 1]);
            //   <--LSB-->
            i -= i & (-i);
        }
        return s;
    }
};

const int M = 26;

struct CharNum {
    int c[M];
};
const CharNum zeros = {0};

int main() {
    int N, Q;
    string S;
    cin >> N >> S >> Q;
    auto add = [](CharNum x, CharNum y) -> CharNum {
        CharNum retval = zeros;
        for(int i = 0; i < M; i++) retval.c[i] = x.c[i] + y.c[i];
        return retval;
    };
    Fenwick<CharNum, decltype(add)> fenwick(N, add, zeros);
    for(int i = 0; i < S.size(); i++) {
        char c = S[i];
        CharNum x = zeros;
        x.c[c - 'a'] = 1;
        fenwick.update(i, x);
    }
    for (int q = 0; q < Q; q++) {
        int mode, i, l, r;
        char c;
        cin >> mode;
        if (mode == 1) {
            cin >> i >> c;
            i--;
            if(S[i] == c) continue;
            CharNum x = zeros;
            x.c[S[i] - 'a'] = -1;
            fenwick.update(i, x);
            x = zeros;
            x.c[c - 'a'] = 1;
            fenwick.update(i, x);
            S[i] = c;
        } else if (mode == 2) {
            cin >> l >> r;
            l--;
            r--;
            CharNum L, R;
            L = l ? fenwick.sum(l - 1) : zeros;
            R = fenwick.sum(r);
            int ans = 0;
            for(int i = 0; i < M; i++) {
                ans += R.c[i] - L.c[i] > 0 ? 1 : 0;
            }
            cout << ans << endl;
        }
    }
}

