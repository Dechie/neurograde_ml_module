#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <cfloat>
#include <map>
#include <utility>
#include <set>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <sstream>
#include <complex>
#include <stack>
#include <queue>
#include <numeric>

using namespace std;
template<typename T>
using reversed_priority_queue = std::priority_queue<T, std::vector<T>, std::greater<T> >;
typedef unsigned long long int ull;
typedef long long int ll;
#define rep(i,a,b) for (ll i=(a); i<(b); i++)
typedef pair<int,int> PII;
typedef pair<int,pair<int,int> > PIII;
typedef pair<ll,ll> PLL;
typedef vector<ll> VL;
typedef vector<VL> MATRIX;
typedef vector<char> VC;
typedef vector<VC> CMATRIX;

ll mpow(ll x, ll y, ll m) {
    x %= m;
    ll result = 1;
    while (y > 0) {
        if (y & 1) result = (result * x) % m;
        x = (x * x) % m;
        y >>= 1;
    }
    return result;
}

// mod inverse 法mにおけるxの逆数
ull minverse(ull x, ull m) {
    return mpow(x, m-2, m);
}


// 組合せ O(n)
ll c (ll n, ll k, ll m) {
    static ll memo[1000][1000];
    if (memo[n][k]!=0) return memo[n][k];
    else if(n==k)      return memo[n][k]=1;
    else if(k==0)      return memo[n][k]=1;
    else               return memo[n][k]=(c(n-1,k-1,m)+c(n-1,k,m))%m;
}

// 組合せ O(n+k)
ll c2 (ll n, ll k, ll m){
    if(n==0 && k==0) return 1;
    if(k>n) return 0;
    if(n<=0) return 0;
    if(n==k) return 1;

    ll res=1;
    for(ll x=n-k+1; x<=n; x++){
        res = res*x;
        if(m!=-1) res %= m;
    }
    for(ll x=1; x<=k; x++) {
        if(m==-1) res = res/x;
        else      res = (res*minverse(x,m)) %m;
    }
    return res;
}

// 組み合わせ O(n)の c3_init 実行後，O(1)のc3を実行
ll  fact[400000];
ll rfact[400000];
void c3_init(ll m){
    fact[0]  = rfact[0] = 1;
    for(int i=1; i<400000; i++){
        fact[i] = (fact[i-1]*i)%m;
        rfact[i] = minverse(fact[i],m);
    }
}
ll c3(ll n, ll k, ll m){
    return (((fact[n] * rfact[k])%m) * rfact[n-k])%m;
}


// 重複組合せ O(n+k)
ll hcomp(ll n, ll k, ll m) {
    return c(k+n-1,k,m);
}

// 重複組み合わせ O(n)
ll hcomp2(ll n, ll k, ll m) {
    //return c(k+n-1,k,m);
    ll res=1;
    for (ll j=1; j<n; j++){
        res = (res * (k+j)) % m;
        res = (res * minverse(j, m)) % m;
    }
    return res;
}

//=========== Union find ============================
class UnionFind {
    public:
        int n;
        vector<int> par;
        vector<int> num;

        UnionFind(int n){ // [0 n)のunion find
            this->n = n;
            this->par.resize(n);
            this->num.resize(n,1);
            for(int i=0;i<n;i++){
                this->par[i] = i;
            }
        }

        int root(int i) {
            if(par[i]==i) return i;
            else return par[i]=root(par[i]);
        }

        bool same(int i, int j){
            return root(i)==root(j);
        }

        void unite(int i,int j){
            int t = root(i);
            int k = root(j);

            if(t==k) return;

            par[t] = k;
            num[k] += num[t];
        }

        int size(int i){
            return num[root(i)];
        }
};

//=========== セグメントツリー ============================

template <typename T>
class SegTree {
    private:
        int _n;
        T _init_val;
        vector<T> _data;
        function<T(T,T)> _f;

        T _query(int a, int b, int k, int l, int r) {
            // [a b)と[l r]が交差しない
            if (r<=a || b <=l) return _init_val;

            // [a b)が[l r]を含む
            if (a<=l && r<=b) return _data[k];

            T vl = _query(a,b, k*2+1, l, (l+r)/2);
            T vr = _query(a,b, k*2+2, (l+r)/2, r);

            return _f(vl,vr);
        }

    public:
        SegTree (int n, T init_val, function<T(T, T)> f) {
            _init_val = init_val;
            _n = 1;
            _f = f;

            while(_n < n) _n *= 2;
            _data.resize(2*_n, init_val);
        }

        T get (int k) {
            return _data[k + _n -1];
        }

        void update(int k, T a) {
            k += _n-1;
            _data[k] = a;
            while(k>0){
                k = (k-1)/2;
                _data[k] = _f( _data[k*2+1], _data[k*2+2] );
            }
        }

        // [a b)の範囲でfを適用
        T query(int a,int b) {
            return _query(a,b,0,0,_n);
        }

        // 先頭からチェックし，はじめてxを満たしたインデックス
        int lower_bound (T x) {
            return _lower_bound(x,0);
        }

        int _lower_bound (T x, int k){
            if (k >= _n-1){
                return k- (_n-1); // 葉
            } else if (_data[k] < x) {
                return _n;        // 該当なし
            } else if (x <= _data[k*2+1]){
                return _lower_bound(x,k*2+1);
            } else {
                return _lower_bound(x- _data[k*2+1] ,k*2+2);
            }
        }
};

// (1 + a + a^2 + a^3 ... + a^(n-1) ) % m
ll modpowsum(ll a, ll n, ll m) {
    if(n==0){
        return 0;
    } else if(n%2==1) {
        return (1+modpowsum(a,n-1,m)*a) % m;
    } else{
        ll t = modpowsum(a,n/2,m);
        return (t+t*mpow(a,n/2,m)) % m;
    }
}

// 範囲 1-3, 5-7, 10-199
// ↓ 2-6を追加
// 1-7, 10-199
class Block {
    public:
        vector< pair<int,int> > block;

        void add(int l, int r) {
            // ブロックを生成
            auto it = lower_bound(begin(this->block),end(this->block),make_pair(l,r));
            auto jt = it; jt--;

            //if(it!=block.end()){
            //    cout<<"it: ("<<it->first<<"-"<<it->second<<endl;
            //}

            // 1.後半のブロックとくっつける
            if(it != block.end() && r >= it->first){
                it->first  = min(l,  it->first);
                it->second = max(r, it->second);

                // その後前半のブロックとくっつける
                auto jt = it; jt--;
                if(it!=block.begin() && jt->second >= it->first){
                    it->first = min(it->first, jt->first);
                    it->second = max(it->second, jt->second);
                    this->block.erase(jt);
                }
            }
            // 2.前半のブロックとくっつける
            else if(it!=block.begin() && jt->second >= l){
                jt->first = min(jt->first, l);
                jt->second = max(jt->second, r);
            }
            // 3.くっつかない
            else {
                block.insert(it,make_pair(l,r));
            }
        }
};

// 組み合わせ
    template < class BidirectionalIterator >
bool next_combination ( BidirectionalIterator first1 ,
        BidirectionalIterator last1 ,
        BidirectionalIterator first2 ,
        BidirectionalIterator last2 )
{
    if (( first1 == last1 ) || ( first2 == last2 )) {
        return false ;
    }
    BidirectionalIterator m1 = last1 ;
    BidirectionalIterator m2 = last2 ; --m2;
    while (--m1 != first1 && !(* m1 < *m2 )){
    }
    bool result = (m1 == first1 ) && !(* first1 < *m2 );
    if (! result ) {
        while ( first2 != m2 && !(* m1 < * first2 )) {
            ++ first2 ;
        }
        first1 = m1;
        std :: iter_swap (first1 , first2 );
        ++ first1 ;
        ++ first2 ;
    }
    if (( first1 != last1 ) && ( first2 != last2 )) {
        m1 = last1 ; m2 = first2 ;
        while (( m1 != first1 ) && (m2 != last2 )) {
            std :: iter_swap (--m1 , m2 );
            ++ m2;
        }
        std :: reverse (first1 , m1 );
        std :: reverse (first1 , last1 );
        std :: reverse (m2 , last2 );
        std :: reverse (first2 , last2 );
    }
    return ! result ;
}

    template < class BidirectionalIterator >
bool next_combination ( BidirectionalIterator first ,
        BidirectionalIterator middle ,
        BidirectionalIterator last )
{
    return next_combination (first , middle , middle , last );
}
long long bit_count(unsigned long long i)
{
    i = i - ((i >> 1) & 0x5555555555555555);
    i = (i & 0x3333333333333333) + ((i >> 2) & 0x3333333333333333);
    return (((i + (i >> 4)) & 0x0F0F0F0F0F0F0F0F) * 0x0101010101010101) >> 56;
}

const int dy[] = {-1,0,1,0};
const int dx[] = {0,1,0,-1};

class SuffixArray {
    public:
        string s; // 対象文字列
        vector<int> sa;  // suffix array本体     i -> sa[i]
        vector<int> rsa; // suffix array本体 sa[i] -> i
        vector<int> lcp; // Longest Common Prefix Array，高さ配列
        SegTree<int> st_lcp; // lcpのセグメントツリー

        // O(n log n)
        SuffixArray(string s): s(s), st_lcp(s.size()+1,1<<29,[](int a,int b){return min(a,b);}) {
            sa = construct_sa();
            lcp = construct_lcp();

            // saの逆も作る
            rsa = vector<int>(s.size()+1,0);
            for(int i=0;i<sa.size();i++){
                rsa[sa[i]]=i;
            }

            // セグメントツリーの初期化 O(n log n)
            for(int i=0;i<lcp.size();i++){
                st_lcp.update(i,lcp[i]);
            }
        }

        // (rank[i],rank[i+k])と(rank[j], rank[j+k])を比較
        bool compare_sa(int i, int j, int k, int n, vector<int> &rank){
            if(rank[i] != rank[j]){
                return rank[i] < rank[j];
            }else{
                int ri = i+k<=n ? rank[i+k] : -1;
                int rj = j+k<=n ? rank[j+k] : -1;
                return ri<rj;
            }
        }

        // O(n)
        vector<int> construct_sa(){
            int n = s.size();
            vector<int> res(n+1,0);
            vector<int> rank(n+1,0);
            vector<int> tmp(n+1,0);

            // 最初は，文字コードをランクにする
            for (int i=0; i<=n; i++){
                res[i] = i;
                rank[res[i]] = i<n ? s[i] : -1;
            }

            // k文字についてソートされているところから，2k文字でソートする
            for(int k=1;k<=n;k*=2){
                auto f=  [&](int i, int j){ return this->compare_sa(i,j,k,n,rank);};
                sort(begin(res),end(res),f);
                tmp[res[0]] = 0;
                for(int i=1;i<=n;i++){
                    tmp[res[i]] = tmp[res[i-1]] + (f(res[i-1],res[i])?1:0);
                }
                rank=tmp;
            }

            return res;
        }

        // O(n)
        // 高さ配列lcpを作成
        vector<int> construct_lcp(){
            int n = s.size();
            vector<int> rank(n+1,0);
            vector<int> res(n+1,0);
            for(int i=0;i<=n;i++) rank[sa[i]]=i;

            int h = 0;
            res[0]=0;
            for(int i=0;i<n;i++){
                int j = sa[rank[i]-1];
                if(h>0) h--;
                for(; j+h<n && i+h<n; h++){
                    if(s[j+h] != s[i+h])
                        break;
                }

                res[rank[i]-1]=h;
            }
            return res;
        }

        void dump() {
            int n= s.size();
            cout<<"# SuffixArray dump"<<endl;
            cout<<"s = "<<s<<endl;
            cout<<"----------------------"<<endl;
            cout<<" i sa[i] lcp[i]  s[sa[i]...]"<<endl;
            for(int i=0;i<=n;i++){
                printf("%2d%6d%7d  %s\n",i,sa[i],lcp[i],s.substr(sa[i]).c_str());
            }
        }

        // O(1)
        // 部分文字列s[i...]のランクを返す
        int getRank(int i){
            return rsa[i];
        }

        // O(log n)
        // suffix array を使って部分文字列patのs内の位置を検索
        // 見つかれなければ-1を返す
        int find(const string &pat){
            int a=0, b=s.size();
            while(b-a>1){
                int c = (a+b)/2;
                if(s.compare(sa[c], pat.size(), pat) < 0) a = c;
                else b = c;
            }

            if(s.compare(sa[b],pat.size(),pat)==0){
                return sa[b];
            }else{
                return -1;
            }
        }

        // 文字列sの s[i..] と s[j..] の先頭共通文字数を返す
        // O(log n)
        int prefixCommonLength(int i, int j){
            if(i==j)  return s.size()-i;

            i = rsa[i];
            j = rsa[j];

            return st_lcp.query(min(i,j), max(i,j));
        }
};

// s % m
ll strmod (const string s, const ll m){
    if(m==1)
        return 0;

    ll len=s.size(), res=0;
    ll t=1; // t: 10^i mod m
    rep(i,0,len){
        int x = s[len-i-1] - '0';
        res = (res + x * t) % m;
        t = (t*10) % m;
    }
    return res;
}

// 最長部分文字列
ll LCS(string s, string t){
    ll s_len = s.size();
    ll t_len = t.size();

    // テーブルの初期化
    ll table[s_len+1][t_len+1];
    rep(i,0,s_len+1) table[i][0]=0;
    rep(j,0,t_len+1) table[0][j]=0;

    rep(i,0,s_len)
        rep(j,0,t_len)
        if(s[i]==t[j]){
            table[i+1][j+1] = max(max(table[i+1][j], table[i][j+1]), table[i][j]+1);
        }else{
            table[i+1][j+1] = max(max(table[i+1][j], table[i][j+1]), table[i][j]);
        }
    return table[s_len][t_len];
}
//============================================
PLL f(PLL x, PLL y){
    return min(x,y);
}

const ll INF=1L<<59;
ll W,H;
SegTree<PLL> xSeg(100100,{INF,INF},f); // {p_i,i}
SegTree<PLL> ySeg(100100,{INF,INF},f); // {q_i,i}

//// X座標が(a,b)，Y座標が(c,d)の矩形における最小コスト
//ll solve(ll a, ll b, ll c, ll d, ll W, ll H){
//    if(a==b && c==d){
//        return 0;
//    }
//    auto xMin = xSeg.query(a,b);
//    auto yMin = ySeg.query(c,d);
//    ll ans=0;
//    if(xMin < yMin){ // 横に結合
//    }else{ // 縦に結合
//        ans = solve()
//    }
//    return ans;
//}
ll solve(ll w, ll h) {
    auto xMin = xSeg.query(0,W);
    auto yMin = ySeg.query(0,H);
    if(xMin.first == INF && yMin.first == INF){ // 辺なし
        return 0;
    } else if(xMin < yMin){ // 横結合
        ll cost = xMin.first;
        xSeg.update( xMin.second, {INF,INF});
        return xMin.first*(h+1) + solve(w-1,h);
    }else{ // 縦結合
        ll cost = yMin.first;
        ySeg.update( yMin.second, {INF,INF});
        return yMin.first*(w+1) + solve(w,h-1);
    }
}

signed main() {
    cin>>W>>H;
    rep(i,0,W){ ll p; cin>>p; xSeg.update(i,{p,i}); }
    rep(i,0,H){ ll q; cin>>q; ySeg.update(i,{q,i}); }
    //cout<<solve(0,W,0,H)<<endl;
    cout<<solve(W,H)<<endl;
    return 0;
}
