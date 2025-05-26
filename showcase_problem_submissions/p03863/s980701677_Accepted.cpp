#include<bits/stdc++.h>

using namespace std;
const int N = 1e5 + 5;
typedef long long ll;

int cnt[30] = {0};
char str[N];
bool up[30];

int main() {
    scanf("%s", str);
    int len = strlen(str);
    for(int i = 0; i < len; i++) {
        up[str[i] - 'a'] = true;
        cnt[str[i] - 'a']++;
    }

    int tp = 0;
    for(int i = 0; i < 26; i++) {
        if(up[i]) tp++;
    }

    if(tp == 2) {
        printf("Second");
        return 0;
    }

    int remain = len - 2;
    if(len & 1) {
        if(str[0] == str[len - 1]) printf("Second");
        else printf("First");
    }
    else {
        if(str[0] == str[len  - 1]) printf("First");
        else printf("Second");
    }

    return 0;
}
