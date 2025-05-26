import java.util.Scanner;

public class Main {
	static int MOD = 1000000007;

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);

		int n = sc.nextInt();
		int m = sc.nextInt();
		sc.nextLine();
		int[] s = new int[n];
		for (int i = 0; i < n; i++) {
			s[i] = sc.nextInt();
		}
		int[] t = new int[m];
		for (int i = 0; i < m; i++) {
			t[i] = sc.nextInt();
		}

		int[][] sum = new int[s.length + 1][t.length + 1];

		for (int i = 1; i <= s.length; i++) {
			for (int j = 1; j <= t.length; j++) {
				sum[i][j] = (sum[i - 1][j] + sum[i][j - 1]) % MOD;
				sum[i][j] = (sum[i][j] - sum[i - 1][j - 1]) % MOD;
				if (s[i - 1] == t[j - 1]) {
					sum[i][j] = (sum[i][j] + sum[i - 1][j - 1] + 1) % MOD;
				}
			}
		}

		System.out.println(sum[s.length][t.length] + 1);
	}
}
