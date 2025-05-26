import java.util.*;

public class Main {
	static final long MOD = 1000000007;
	public static void main (String[] args) {
		Scanner sc = new Scanner(System.in);
		char[] line = sc.next().toCharArray();
		long[] dp1 = new long[line.length]; // L >= a + b
		long[] dp2 = new long[line.length];	// L = a + b
		dp1[0] = 3;
		dp2[0] = 2;
		for (int i = 1; i < line.length; i++) {
			if (line[i] == '0') {
				dp1[i] = (dp1[i - 1] * 3 - dp2[i - 1] * 2 + MOD) % MOD;
				dp2[i] = dp2[i - 1];
			} else {
				dp1[i] = dp1[i - 1] * 3 % MOD;
				dp2[i] = dp2[i - 1] * 2 % MOD;
			}
		}
		System.out.println(dp1[dp1.length - 1]);
	}
}
