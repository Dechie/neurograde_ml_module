import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		final int MAX = 1000000;

		Scanner sc = new Scanner(System.in);

		boolean primes[] = new boolean[MAX];
		primes[1] = true;
		for (int i = 2; i * i < MAX; i++) {
			for (int j = 1; i * j < MAX; j++) {
				primes[i * j] = true;
			}
		}

		while (true) {
			int a = sc.nextInt();
			int d = sc.nextInt();
			int n = sc.nextInt();
			if (a == 0)
				break;
			while (n != 0) {
				if (!primes[a])
					n--;
				a += d;
			}
			System.out.println(a - d); // while??§?¶??????????????????§
		}
	}
}