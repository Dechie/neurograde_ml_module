import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		new Main().run();
	}

	final long MODULO = 1_000_000_000 + 7;

	public void run() {
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
		long[] a = new long[n];
		for (int i = 0; i < n; ++i) {
			a[i] = sc.nextLong();
		}
		boolean f = true;
		long ans = 0;
		while (f) {
			f = false;
			out: for (int i = 0; i < n; ++i) {
				if (a[i] > n - 1) {
					long d = (a[i] - (n - 1) + n - 1) / n;
					ans += d;
					a[i] -= d * n;
					for (int j = 0; j < n; ++j) {
						if (i == j)
							continue;
						a[j] += d;
						f = true;
					}
					break out;
				}
			}
		}
		System.out.println(ans);
	}

	long f(long[] a, long k) {
		int n = a.length;
		long s = 0;
		for (int i = 0; i < n; ++i) {
			if (a[i] + k <= n - 1) {
				continue;
			} else {
				long res = a[i] + k - (n - 1);
				s += (res + n) / (n + 1);
			}
		}
		return k - s;

	}

	static void tr(Object... objects) {
		System.out.println(Arrays.deepToString(objects));
	}
}
