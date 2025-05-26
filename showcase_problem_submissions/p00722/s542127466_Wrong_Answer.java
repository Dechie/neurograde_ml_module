

import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);

		boolean[] table = new boolean[10000000];

		for (int i = 2; i < 10000000; i++) {
			table[i] = true;
		}
		table[0] = false;
		table[1] = false;
		for (int i = 2; i < 10000000; i++) {
			if (table[i]) {
				for (int j = i + i; j < 10000000; j += i)
					table[j] = false;
			}
		}

		while (true) {
			int a = scan.nextInt();
			int d = scan.nextInt();
			int n = scan.nextInt();

			int count = 0;
			int ans = 0;

			if (a == 0 && d == 0 && n == 0)
				break;

			for (int i = a; i < 10000000; i += d) {
				if (table[i])
					count++;
				if (n == count) {
					ans = i;

				}

			}
			System.out.println(ans);
		}

		scan.close();
	}
}