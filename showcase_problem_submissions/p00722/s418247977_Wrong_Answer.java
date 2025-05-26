import java.util.Scanner;

class Main {
	static int suuretuare = 0;
	public static void main(String[] args) {
		Scanner stdIn = new Scanner(System.in);
		
		int n = stdIn.nextInt();
		int k = stdIn.nextInt();
		int j = stdIn.nextInt();
		int[] a = new int[j];
		for (int i = 0; suuretuare < j; i++) {
			 int tmp = n + (i * k);
			 if (sieve(tmp)) {
				 a[suuretuare] = tmp;
				 suuretuare++;
			 }
		}
		System.out.println(a[j-1]);
		
	}
	
	static boolean sieve(int p) {
		int tmp = p;
		if (tmp == 1) {
			return false;
		}
		for (int i = 2; i <= tmp-1; i++) {
			if (tmp % i == 0) {
				return false;
			}
		}
		return true;
	}
}