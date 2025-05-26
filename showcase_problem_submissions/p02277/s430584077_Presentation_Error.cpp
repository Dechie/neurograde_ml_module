#include <iostream>

using namespace std;

char S1[100000], S2[100000];

template<typename T>
void swap(T* a, T* b){
	T tmp = *a;
	*a = *b;
	*b = tmp;
}

void bubble(int A[], int n){
	int flag;
	while(flag){
	flag = false;
	for(int i = n-1;i >= 1;--i){
		if(A[i] < A[i-1]){
			swap(&A[i], &A[i-1]);
			swap(&S1[i], &S1[i-1]);
			flag = true;
			}
		}
	}
}

int partition(int A[], int p, int r){
	int x = A[r], cnt = p-1;
	for(int i=p;i<r;++i){
		if(A[i] <= x){
			cnt++;
			swap(&A[cnt], &A[i]);
			swap(&S2[cnt], &S2[i]);
		}
	}
	swap(&A[cnt+1], &A[r]);
	swap(&S2[cnt+1], &S2[r]);
	return ++cnt;
}

void quickSort(int A[], int p, int r){
	if(p < r){
		int q = partition(&A[0], p, r);
		quickSort(&A[0], p, q-1);
		quickSort(&A[0], q+1, r);
	}
}

void judge(int n){
	int flag = true;
	for(int i=0;i<n;++i){
		if(S1[i] != S2[i]){
			flag = false;
			break;
		}
	}
	if(flag)cout << "Stable" << endl;
	else cout << "Not stable" << endl;
}

int main(){
	int n, A1[100000], A2[100000];
	cin >> n;
	for(int i=0;i<n;++i){
		cin >> S1[i] >> A1[i];
		S2[i] = S1[i], A2[i] = A1[i];
	}
	bubble(&A1[0], n);
	quickSort(&A2[0], 0, n-1);
	judge(n);
	
	for(int i=0;i<n;++i)
		cout << S2[i] << " " << A2[i] << endl;
	cout << "\n";
	return 0;
}
