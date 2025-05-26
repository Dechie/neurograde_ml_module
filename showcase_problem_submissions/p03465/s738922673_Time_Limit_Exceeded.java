
import java.util.Arrays;

public class Main {

	static int n;
	static int[] a;
	private static void solve() {
		
		n = ni();
		a = na(n);

		int sum = Arrays.stream(a).sum();


		int[][] dp = new int[2][sum + 1];
		dp[0][0] = 1;

		for (int i = 0; i < n; i ++) {
			int from = i % 2;
			int to = (i + 1) % 2;
			
			for (int j = 0; j <= sum; j ++) {
				dp[to][j] = dp[from][j];
				if (j >= a[i]) {
					dp[to][j] += dp[from][j - a[i]];
				}
				if (dp[to][j] > 2) {
					dp[to][j] = 2;
				}
			}
		}
		
		int ret = 0;
		int[] arr = dp[n % 2];

		if (sum % 2 == 1) {
			for (int i = (sum + 1) / 2;; i ++) {
				if (arr[i] >= 1) {
					ret = i;
					break;
				}
			}
		} else {
			if (arr[sum / 2] > 1) {
				ret = sum / 2;
			} else {
				for (int i = sum / 2 + 1;; i ++) {
					if (arr[i] >= 1) {
						ret = i;
						break;
					}
				}
			}
		}
		
		
		//System.out.println(Arrays.toString(dp[n % 2]));
		System.out.println(ret);
	}
	

	public static void main(String[] args) {
		new Thread(null, new Runnable() {
			@Override
			public void run() {
				long start = System.currentTimeMillis();
				String debug = args.length > 0 ? args[0] : null;
				if (debug != null) {
					try {
						is = java.nio.file.Files.newInputStream(java.nio.file.Paths.get(debug));
					} catch (Exception e) {
						throw new RuntimeException(e);
					}
				}
				reader = new java.io.BufferedReader(new java.io.InputStreamReader(is), 32768);
				solve();
				out.flush();
				tr((System.currentTimeMillis() - start) + "ms");
			}
		}, "", 64000000).start();
	}

	private static java.io.InputStream is = System.in;
	private static java.io.PrintWriter out = new java.io.PrintWriter(System.out);
	private static java.util.StringTokenizer tokenizer = null;
	private static java.io.BufferedReader reader;

	public static String next() {
		while (tokenizer == null || !tokenizer.hasMoreTokens()) {
			try {
				tokenizer = new java.util.StringTokenizer(reader.readLine());
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		}
		return tokenizer.nextToken();
	}

	private static double nd() {
		return Double.parseDouble(next());
	}

	private static long nl() {
		return Long.parseLong(next());
	}

	private static int[] na(int n) {
		int[] a = new int[n];
		for (int i = 0; i < n; i++)
			a[i] = ni();
		return a;
	}

	private static char[] ns() {
		return next().toCharArray();
	}

	private static long[] nal(int n) {
		long[] a = new long[n];
		for (int i = 0; i < n; i++)
			a[i] = nl();
		return a;
	}

	private static int[][] ntable(int n, int m) {
		int[][] table = new int[n][m];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				table[i][j] = ni();
			}
		}
		return table;
	}

	private static int[][] nlist(int n, int m) {
		int[][] table = new int[m][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				table[j][i] = ni();
			}
		}
		return table;
	}

	private static int ni() {
		return Integer.parseInt(next());
	}

	private static void tr(Object... o) {
		if (is != System.in)
			System.out.println(java.util.Arrays.deepToString(o));
	}
}
