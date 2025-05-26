import java.io.*;
import java.lang.reflect.Array;
import java.math.*;
import java.util.*;

public class Main {
	static final int INF = (int) (1e9 + 10);
	static final long MOD = (int) (1e9 + 7);
	static final int N = (int) (4e5 + 5);
	// static ArrayList<Integer>[] graph;
	// static boolean visited[];
	// static long size[];

	public static void main(String[] args) throws NumberFormatException, IOException {
		FastReader sc = new FastReader();
		PrintWriter pr = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		// Scanner scn = new Scanner(System.in);
		//
		int a = sc.nextInt();
		int b = sc.nextInt();
		int c = sc.nextInt();
		int d = sc.nextInt();
		long dp[][] = new long[c + 1][d + 1];
		dp[a][b]=1;
		for (int i = a; i <= c; i++) {
			for (int j = b; j <= d; j++) {
				if (i == a && j == b)
					continue;
					dp[i][j] = ((dp[i - 1][j] * j) +( dp[i][j - 1] * i) - (dp[i - 1][j - 1]*(i-1)*(j-1))+998244353)%998244353;
		//System.out.println(dp[i - 1][j]+" "+dp[i][j - 1]);
			}
		}
		pr.println(dp[c][d]%998244353);
		//
		// Coded to Perfection by Rohan Mukhija
		pr.flush();
		pr.close();
	}

	/*
	 * Template From Here
	 */
	private static boolean possible(long[] arr, long mid, long k, long t) {
		long c = 1;
		long tot = 0;
		for (int i = 0; i < arr.length; i++) {
			tot += (arr[i]);
			if (tot > mid) {
				tot = arr[i];
				c++;
			}
		}
		// System.out.println(mid+" "+c);
		if (c <= k)
			return true;
		return false;
	}

	public static class Pair implements Comparable<Pair> {
		int u;
		int v;

		Pair(int u, int v) {
			this.u = u;
			this.v = v;

		}

		public int compareTo(Pair compareEdge) {
			return Long.compare(this.v, compareEdge.v);
		}
	};

	static void sort(long[] a) {
		ArrayList<Long> l = new ArrayList<>();
		for (long i : a)
			l.add(i);
		Collections.sort(l);
		for (int i = 0; i < a.length; i++)
			a[i] = l.get(i);
	}

	static void sort(int[] a) {
		ArrayList<Integer> l = new ArrayList<>();
		for (int i : a)
			l.add(i);
		Collections.sort(l);
		for (int i = 0; i < a.length; i++)
			a[i] = l.get(i);
	}

	public static int lowerBound(ArrayList<Integer> array, int length, long value) {
		int low = 0;
		int high = length;
		while (low < high) {
			final int mid = (low + high) / 2;
			if (value <= array.get(mid)) {
				high = mid;
			} else {
				low = mid + 1;
			}
		}
		return low;
	}

	static class FastReader {
		BufferedReader br;
		StringTokenizer st;

		public FastReader() {
			br = new BufferedReader(new InputStreamReader(System.in));
		}

		String next() {
			while (st == null || !st.hasMoreElements()) {
				try {
					st = new StringTokenizer(br.readLine());
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			return st.nextToken();
		}

		int nextInt() {
			return Integer.parseInt(next());
		}

		long nextLong() {
			return Long.parseLong(next());
		}

		double nextDouble() {
			return Double.parseDouble(next());
		}

		String nextLine() {
			String str = "";
			try {
				str = br.readLine();
			} catch (IOException e) {
				e.printStackTrace();
			}
			return str;
		}

		public int[] nextIntArray(int n) {
			int[] a = new int[n];
			for (int i = 0; i < n; i++)
				a[i] = nextInt();
			return a;
		}

		public long[] nextLongArray(int n) {
			long[] a = new long[n];
			for (int i = 0; i < n; i++)
				a[i] = nextLong();
			return a;
		}
	}

	static long gcd(long a, long b) {
		if (a == 0)
			return b;
		return gcd(b % a, a);
	}

	static long lcm(long a, long b) {
		return (a * b) / gcd(a, b);
	}

	public static void sortbyColumn(int[][] att, int col) {
		// Using built-in sort function Arrays.sort
		Arrays.sort(att, new Comparator<int[]>() {

			@Override
			// Compare values according to columns
			public int compare(final int[] entry1, final int[] entry2) {

				// To sort in descending order revert
				// the '>' Operator
				// if (entry1[col] > entry2[col])
				// return 1;
				// else //(entry1[col] >= entry2[col])
				// return -1;
				return new Integer(entry1[col]).compareTo(entry2[col]);
				// return entry1[col].
			}
		}); // End of function call sort().
	}

	public static void sortbyColumn(double[][] att, int col) {
		// Using built-in sort function Arrays.sort
		Arrays.sort(att, new Comparator<double[]>() {

			@Override
			// Compare values according to columns
			public int compare(final double[] entry1, final double[] entry2) {

				// To sort in descending order revert
				// the '>' Operator
				// if (entry1[col] > entry2[col])
				// return 1;
				// else //(entry1[col] >= entry2[col])
				// return -1;
				return new Double(entry1[col]).compareTo(entry2[col]);
				// return entry1[col].
			}
		}); // End of function call sort().
	}

	static class pair {
		int V, I;

		pair(int v, int i) {
			V = v;
			I = i;
		}
	}

	public static int[] swap(int data[], int left, int right) {
		int temp = data[left];
		data[left] = data[right];
		data[right] = temp;
		return data;
	}

	public static int[] reverse(int data[], int left, int right) {
		while (left < right) {
			int temp = data[left];
			data[left++] = data[right];
			data[right--] = temp;
		}
		return data;
	}

	public static boolean findNextPermutation(int data[]) {
		if (data.length <= 1)
			return false;

		int last = data.length - 2;
		while (last >= 0) {
			if (data[last] < data[last + 1]) {
				break;
			}
			last--;
		}
		if (last < 0)
			return false;

		int nextGreater = data.length - 1;
		for (int i = data.length - 1; i > last; i--) {
			if (data[i] > data[last]) {
				nextGreater = i;
				break;
			}
		}
		data = swap(data, nextGreater, last);
		data = reverse(data, last + 1, data.length - 1);
		return true;
	}

	static long ncr(long a, long b) {
		if (b > a - b)
			return ncr(a, a - b);
		long ansMul = 1;
		long ansDiv = 1;
		for (int i = 0; i < b; i++) {
			ansMul *= (a - i);
			ansDiv *= (i + 1);
			ansMul %= MOD;
			ansDiv %= MOD;
		}

		long ans = ansMul * power(ansDiv, MOD - 2, MOD) % MOD;

		return ans;
	}

	static long __gcd(long n1, long n2) {
		long gcd = 1;

		for (int i = 1; i <= n1 && i <= n2; ++i) {
			// Checks if i is factor of both integers
			if (n1 % i == 0 && n2 % i == 0) {
				gcd = i;
			}
		}
		return gcd;
	}

	static long power(long prev, long n2, long mod2) {
		long res = 1;

		prev = prev % mod2;

		if (prev == 0)
			return 0;
		while (n2 > 0) {

			if ((n2 & 1) == 1)
				res = (res * prev) % mod2;

			n2 = n2 >> 1;
			prev = (prev * prev) % mod2;
		}
		return res;
	}

	static long modInverse(long fac2, int p) {
		return power(fac2, p - 2, p);
	}

	static boolean isPrime(int n) {
		// Corner cases
		if (n <= 1)
			return false;
		if (n <= 3)
			return true;

		// This is checked so that we can skip
		// middle five numbers in below loop
		if (n % 2 == 0 || n % 3 == 0)
			return false;

		for (int i = 5; i * i <= n; i = i + 6)
			if (n % i == 0 || n % (i + 2) == 0)
				return false;

		return true;
	}

	public static BigInteger lcmm(String a, String b) {
		BigInteger s = new BigInteger(a);
		BigInteger s1 = new BigInteger(b);
		BigInteger mul = s.multiply(s1);
		BigInteger gcd = s.gcd(s1);
		BigInteger lcm = mul.divide(gcd);
		return lcm;
	}
	/*
	 * static boolean prime[] = new boolean[10000007]; static int spf[] = new
	 * int[10000007];
	 * 
	 * public static void sieveOfEratosthenes(int n) { for (int i = 2; i < n;
	 * i++) prime[i] = true; for (int p = 2; p * p <= n; p++) {
	 * 
	 * if (prime[p] == true) {
	 * 
	 * for (int i = p * p; i <= n; i += p) { prime[i] = false; spf[i] = p; } } }
	 * }
	 */
}
