import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
		long[] a =new long[n];
		for(int i=0;i<n;i++) {
			a[i]=sc.nextLong();
		}

		System.out.println(binarySearch(a));


	}

	static long binarySearch(long[] input) {
		int n=input.length;
		long max = Long.MAX_VALUE;
		long min =0;
		while (max>min) {
			long mid=min+(max-min)/2;
			if(done(input,mid)) {
				max=mid;
			} else if (nearDone(input,mid)) {
				for(long i = Math.max(0,mid-n); i<=mid; i++) {
					if(done(input,i)) {
						max=i;
						break;
					}
					if(i==mid) {
						min=mid;
					}
				}


			} else {
				min=mid;
			}
		}
		return min;
	}

	static boolean done(long[]input, long m) {
		int n=input.length;
		long count=0;
		for(int i=0;i<n;i++) {
			count+=(input[i]+m-(n-1)+n)/(n+1);
		}
		return count <= m;
	}

	static boolean nearDone(long[] input, long m) {
		int n=input.length;
		long count=0;
		for(int i=0;i<n;i++) {
			count+=(input[i]+m-(n+n-1)+n)/(n+1);
		}
		return count <= m;
	}
}
