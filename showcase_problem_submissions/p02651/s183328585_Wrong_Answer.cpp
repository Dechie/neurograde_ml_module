#include  <iostream>
#include  <cstdlib>

void solve(void);

int main(void)
{
	solve();

	return EXIT_SUCCESS;
}

void solve(void)
{
	int T;
    int N[100];
    unsigned long long A[100][200];
	std::string S[100];

	std::cin >> T;

	for(int k = 0; k < T; k++){
    	std::cin >> N[k];
    
    	for(int i = 0; i < N[k]; i++)
    		std::cin >> A[k][i];
    
    	std::cin >> S[k]; 
	}

	int x[T] = {0};
	for(int k = 0; k < T; k++){
		for(int i = 0; i < N[k]; i++){
			if(S[k][i] == '0'){
				// x = 0となるように操作をする
				if((x[k] ^ A[k][i]) == 0)
					x[k] ^= A[k][i];
			}
			else if(S[k][i] == '1'){
				// x != 0となるように操作をする
				if((x[k] ^ A[k][i]) != 0)
					x[k] ^= A[k][i];
			}
			else{
			}
		}
	}

	// 回答
	for(int k = 0; k < T; k++){
		int ans = 0;

		if(x[k] == 0)
			ans = 0;
		else
			ans = 1;

		std::cout << ans << std::endl;
	}
}

