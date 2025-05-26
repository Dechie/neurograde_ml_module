import java.util.Scanner;

class Card{
    public String suit;
    public int value;
}

public class Main{
    public static void main(String[] args){
	int i, q;

	int stable = 1;
	
	Scanner s = new Scanner(System.in);
	int n = s.nextInt();
    	Card[] A = new Card[n];
	Card[] B = new Card[n];

	for(i=0; i<n; i++){
	    A[i] = new Card();
	    B[i] = new Card();
	}

	for(i=0; i<n; i++){
	    A[i].suit = B[i].suit = s.next();
	    A[i].value = B[i].value = s.nextInt();
	}

	mergeSort(B, n, 0, n);
	quickSort(A, n, 0, n-1);

	for(i=0; i<n; i++){  
	    //????????????????????¨??????????????????????????????????????????
	    if(A[i].suit != B[i].suit) stable = 0;
	}

	if(stable == 1){
	    System.out.print("Stable\n");
	}else{
	    System.out.print("Not stable\n");
	}
	for(i=0; i<n; i++){
	    System.out.printf("%s %d\n", A[i].suit, A[i].value);
	    
	}
    }
    
    public static int partition(Card[] A, int n, int p, int r){
	int x, i, j;
	Card tmp;
	x = A[r].value;
	i = p-1;
	for(j=p; j<r; j++){
	    if(A[j].value<=x){
		i++;
		tmp = A[i]; A[i] = A[j]; A[j] = tmp;
	    }
	}
	tmp = A[i+1]; A[i+1] = A[r]; A[r] = tmp;
	return i+1;
    }

    public static void quickSort(Card[] A, int n, int p, int r){
	int q;
	if(p < r){
	    q = partition(A, n, p, r);
	    quickSort(A, n, p, q - 1);
	    quickSort(A, n, q + 1, r); 
	}
    }

    public static void merge(Card A[], int n, int left, int mid, int right){
	int i, j, k;
	int SENTINEL=2000000000;
	Card[] L = new Card[100000 / 2+2];
	Card[] R = new Card[100000 / 2+2];

	for(i=0; i<n; i++){
	    L[i] = new Card();
	    R[i] = new Card();
	}

	int n1 = mid - left;
	int n2 = right - mid;
	for(i=0; i<n1; i++) L[i]= A[left + i];
	for(i=0; i<n2; i++) R[i]= A[mid + i];
	L[n1].value = R[n2].value = SENTINEL;
	i = j = 0;
	for(k=left; k<right; k++){
	    if(L[i].value <= R[j].value){
		A[k] = L[i++];
	    }else{
		A[k] = R[j++];
	    }
	}
    }
    public static void mergeSort(Card A[], int n, int left, int right){   
	int mid;
	if(left+1<right){ 
	    mid  = (left + right) / 2;
	    mergeSort(A, n, left, mid);  
	    mergeSort(A, n, mid, right);
	    merge(A, n, left, mid, right);
	}
    }
}