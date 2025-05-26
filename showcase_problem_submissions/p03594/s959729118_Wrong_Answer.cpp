#include <bits/stdc++.h> 
 
using namespace std;
 
template<class T> inline string toString(T x) {ostringstream sout;sout<<x;return sout.str();}

 
//typedef
//------------------------------------------
typedef pair<int, int> PII;
typedef pair<long, long> PLL;
typedef long long LL;

//container util
//------------------------------------------
#define PB emplace_back
#define MP make_pair 
#define SZ(a) int((a).size())
//repetition
//------------------------------------------
#define FOR(i,a,b) for(LL i=(a);i<(b);++i)
#define REP(i,n)  FOR(i,0,n)
#define SORT(c) sort((c).begin(),(c).end())
//constant
//--------------------------------------------
//clear memory
#define CLR(a) memset((a), 0 ,sizeof(a))
 
const LL INF=LONG_MAX;
const double EPS=0.00001;
const long double PI = acos(-1.0);

const int MAX_N=2e5+2;

int main(){
    int H,W,d;
    cin>>H>>W>>d;
    char c[2][2];
    string tmp="RYGB";
    c[0][0]=tmp[0];
    c[0][1]=tmp[1];
    c[1][0]=tmp[2];
    c[1][1]=tmp[3];
    int cnt=0;
    if(d%2==1){
        REP(h,H){
            REP(w,W){
                cout<<c[h%2][w%2];
            }
            cout<<endl;
        }
    }else{
        char res[510][510];
        int H2=H+(4-H%4);
        int W2=W+(4-W%4);
        string tmps[4];
        tmps[0]="RYGB";
        tmps[1]="GBRY";
        tmps[2]="YRBG";
        tmps[3]="BGYR";
        REP(hb,H2){
            REP(wb,W2/4){
                REP(i,4)res[hb][wb*4+i]=tmps[hb%4][i];
            }
        }
        REP(h,H){
            REP(w,W){
                cout<<res[h][w];
            }
            cout<<endl;
        }
    }
    return 0;
}
