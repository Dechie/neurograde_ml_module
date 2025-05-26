import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.math.BigInteger;

public class Main {
	public static void main(String[] args) throws Exception {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		int n = Integer.parseInt(br.readLine());
		String[] sa = br.readLine().split(" ");
		int[] a = new int[n];
		for (int i = 0; i < n; i++) {
			a[i] = Integer.parseInt(sa[i]);
		}
		br.close();

		int m = 2000;
		int m2 = m * m;
		BigInteger[] dp = new BigInteger[m2 + 1];
		dp[0] = BigInteger.ONE;
		for (int i = 1; i <= m2; i++) {
			dp[i] = BigInteger.ZERO;
		}
		for (int i = 0; i < n; i++) {
			for (int j = i * m; j >= 0; j--) {
				dp[j + a[i]] = dp[j + a[i]].add(dp[j]);
			}
		}

		BigInteger k = BigInteger.valueOf(2).pow(n - 1);
		BigInteger sum = BigInteger.ZERO;
		for (int i = 0; i <= m2; i++) {
			sum = sum.add(dp[i]);
			if (sum.compareTo(k) > 0) {
				System.out.println(i);
				return;
			}
		}
	}
}
