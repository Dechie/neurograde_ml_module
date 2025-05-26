import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Arrays;

public class Main {
	static int n;
	static long m;
	static int[] a;

	public static void main(String[] args) throws Exception {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		String[] sa = br.readLine().split(" ");
		n = Integer.parseInt(sa[0]);
		m = Long.parseLong(sa[1]);
		sa = br.readLine().split(" ");
		a = new int[n];
		for (int i = 0; i < n; i++) {
			a[i] = Integer.parseInt(sa[i]);
		}
		br.close();

		Arrays.parallelSort(a);

		long ng = 1000000000000000000L;
		long ok = 0;
		long cnt = 0;
		while (ng - ok > 1) {
			long mid = (ok + ng) / 2;
			long ret = count(mid);
			if (ret >= m) {
				ok = mid;
				cnt = ret;
			} else {
				ng = mid;
			}
		}

		long[] b = new long[n + 1];
		for (int i = 0; i < n; i++) {
			b[i + 1] = b[i] + a[i];
		}
		long ans = 0;
		for (int i = 0; i < n; i++) {
			long val2 = ok - a[i];
			int idx = binarySearch(val2 - 0.5);
			ans += (b[n] - b[idx]);
			ans += (long) a[i] * (n - idx);
		}
		ans -= ok * (cnt - m);
		System.out.println(ans);
	}

	static long count(long val) {
		long sum = 0;
		for (int i = 0; i < n; i++) {
			long val2 = val - a[i];
			int idx = binarySearch(val2 - 0.5);
			sum += n - idx;
		}
		return sum;
	}

	static int binarySearch(double val) {
		int ng = -1;
		int ok = n;

		while (ok - ng > 1) {
			int mid = (ok + ng) / 2;
			if (a[mid] >= val) {
				ok = mid;
			} else {
				ng = mid;
			}
		}
		return ok;
	}
}
