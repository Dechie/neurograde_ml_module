#include <iostream>
using namespace std;

int main(){
    long long n;cin>>n;
    long long acc = 0;
    long long max_num;
    for (long long i=1;i<=n;i++){
        acc += i;
        if (acc >= n){
            max_num = i;
            break;
        }
    }
    cout << max_num << endl;
    n -= max_num;

    for (long long i=max_num-1;i>0;i--){
        if (n  == 0){
            return 0;
        }
        if (n - i >= 0){
            cout << i << endl;
            n -= i;
        }
    }
}