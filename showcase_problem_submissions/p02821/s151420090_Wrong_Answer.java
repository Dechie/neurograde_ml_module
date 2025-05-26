import java.util.*;

class Main{

    public static void main(String args[]) throws Exception{
        Main main = new Main();
    }

    public Main(){
        Scanner scanner = new Scanner(System.in);
        int N = scanner.nextInt();
        long M = scanner.nextLong();

        int[] A = new int[N];

        for(int i=0;i<N;i++){
            A[i] = scanner.nextInt();
        }

        Arrays.sort(A);

        int L = -1, R = A[N-1]+A[N-1];
        while(R-L > 1){
            int mid = L+(R-L)/2;

            int num=0;
            for(int i=0;i<N;i++){
                int t = Arrays.binarySearch(A, mid-A[i]);
                num += N - ((t>=0) ?  t : ~t);
            }

            if(num < M)
                R = mid;
            else
                L = mid;
        }


        long[] S = new long[N+1];
        for(int i=0;i<N;i++) S[i+1] = S[i] + A[i];

        int num;
        long count=0, result = 0;
        for(int i=0;i<N;i++){
            int t = Arrays.binarySearch(A, R-A[i]);
            num = ((t>=0) ?  t : ~t);
            count += N-num;
            result += A[i]*(N-num) + S[N] - S[num];
        }

        if(M > count)
            result += L * (M - count);
        else if(M < count)
            result += R * (M - count);

        System.out.println(result);
    }
}
