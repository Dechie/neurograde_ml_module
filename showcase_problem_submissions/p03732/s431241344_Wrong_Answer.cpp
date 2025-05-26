#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

struct Object {
    int w;
    int v;

    bool operator<( const Object& right ) const {
        return v < right.v;
    }
};

int main() {
    int N, W;//個数、強度
    vector<vector<Object> > list(4);

    cin >> N >> W;
    int tem1, tem2;
    for (int i = 0; i < N; i++) {
        cin >> tem1 >> tem2;
        Object temp = {tem1, tem2};
        if(i == 0) list[0].push_back(temp);
        else list[tem1 - list[0][0].w].push_back(temp);
    }

    for(int i = 0; i < 4; i++) sort(list[i].begin(), list[i].end());

    int weiht = 0, sum = 0, largestsum = 0;
    for(unsigned i0 = 0; i0 < list[0].size() + 1; i0++) {
        for(unsigned i1 = 0; i1 < list[1].size() + 1; i1++) {
            for(unsigned i2 = 0; i2 < list[2].size() + 1; i2++) {
                for(unsigned i3 = 0; i3 < list[3].size() + 1; i3++) {
                    sum = 0, weiht = 0;
                    if(!list[0].empty()) for(unsigned j = 0; j < i0; j++) weiht += list[0][0].w, sum += list[0][list[0].size() - 1 - j].v;
                    if(!list[1].empty()) for(unsigned j = 0; j < i1; j++) weiht += list[0][0].w + 1, sum += list[1][list[1].size() - 1 - j].v;
                    if(!list[2].empty()) for(unsigned j = 0; j < i2; j++) weiht += list[0][0].w + 2, sum += list[2][list[2].size() - 1 - j].v;
                    if(!list[3].empty()) for(unsigned j = 0; j < i3; j++) weiht += list[0][0].w + 3, sum += list[3][list[3].size() - 1 - j].v;
                    if(weiht <= W && sum > largestsum) largestsum = sum;
                }
            }
        }
    }
    cout << largestsum << "\n";

    return 0;
}
