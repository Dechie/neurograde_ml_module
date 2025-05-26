import java.util.*;
import java.io.*;

public class Main {
	static int mod = (int) 1e9 + 7;
	static int DX[] = { -1, 0, 1, 0 }, DY[] = { 0, -1, 0, 1 };
	static final int INF = Integer.MAX_VALUE / 3;
	static final long LINF = Long.MAX_VALUE / 3;

	static int n, a[];
	static long m, sum[];
	
	public static void main(String[] args) {
		FastScanner fs = new FastScanner(System.in);
		n = fs.nextInt(); m = fs.nextLong();
		a = fs.nextIntArray(n);
		rsort(a);
		sum = new long[n+1];
		for(int i=0;i<n;i++)sum[i+1] = sum[i] + a[i];
		//価値xの握手をm個作れるような、xの最大値を二分探索で求める
		int l = 0, r = INF;
		while(l+1<r) {
			int c = (l+r)/2;
			if(check(c)) l = c;
			else r = c;
		}
		System.out.println(culcSum(l));
	}
	
	//x以上になる物だけでm個つくる時の和の最大値
	static long culcSum(int x) {
		x ++;
		long cnt = 0;
		long s = 0;
		for(int i=0;i<n;i++) {
			int l = -1, r = n;
			while(l+1<r) {
				int c = (l+r)/2;
				if(a[i] + a[c] >= x)l = c;
				else r = c;
			}
			if(l<i)break;
			cnt += (l - i + 1) * 2 - 1;
			s += (((long)a[i] * (l-i+1) + (sum[l+1] - sum[i]))*2 - a[i] * 2);
		}
		long rem = m - cnt;
		s += (rem * (x-1));
		return s;
	}
	
	static boolean check(int x) {
		long cnt = 0;
		for(int i=0;i<n;i++) {
			int l = -1, r = n;
			while(l+1<r) {
				int c = (l+r)/2;
				if(a[i] + a[c] >= x)l = c;
				else r = c;
			}
			if(l<i)break;
			cnt += (l - i + 1) * 2 - 1;
		}
		if(cnt >= m)return true;
		else return false;
	}
	
	static void rsort(int x[]) {
		Arrays.sort(x);
		int len = x.length;
		for(int i=0;i<len/2;i++) {
			int tmp = x[i];
			x[i] = x[len-1-i];
			x[len-1-i] = tmp;
		}
	}
}

//高速なScanner
class FastScanner {
	private BufferedReader reader = null;
	private StringTokenizer tokenizer = null;

	public FastScanner(InputStream in) {
		reader = new BufferedReader(new InputStreamReader(in));
		tokenizer = null;
	}

	public String next() {
		if (tokenizer == null || !tokenizer.hasMoreTokens()) {
			try {
				tokenizer = new StringTokenizer(reader.readLine());
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		return tokenizer.nextToken();
	}

	public String nextLine() {
		if (tokenizer == null || !tokenizer.hasMoreTokens()) {
			try {
				return reader.readLine();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		return tokenizer.nextToken("\n");
	}

	public long nextLong() {
		return Long.parseLong(next());
	}

	public int nextInt() {
		return Integer.parseInt(next());
	}

	public double nextDouble() {
		return Double.parseDouble(next());
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