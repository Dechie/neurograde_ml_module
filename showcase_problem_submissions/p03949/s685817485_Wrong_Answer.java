import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.InputMismatchException;
import java.util.NoSuchElementException;

public class Main implements Runnable {
	public static final int INF = 1 << 29;

	public static void main(String[] args) {
		new Thread(null, new Main(), "", 16L * 1024 * 1024).start();
	}
	
	int n;
	ArrayList<Integer>[] tree;
	int k;
	int[] v,p;
	int[] ub,lb,ans;
	
	@SuppressWarnings("unchecked")
	public void run() {
		IO io = new IO();
		n = io.nextInt();
		ub = new int[n];
		lb = new int[n];
		tree = new ArrayList[n];
		for(int i=0;i<n;i++) {
			tree[i] = new ArrayList<>();
		}
		for(int i=0;i<n-1;i++) {
			int a = io.nextInt() - 1;
			int b = io.nextInt() - 1;
			tree[a].add(b);
			tree[b].add(a);
		}
		boolean path = true;
		for(int i=0;i<n;i++) {
			if (tree[i].size() > 2) {
				path = false;
			}
		}
		if (path) throw new RuntimeException();
		k = io.nextInt();
		v = new int[k];
		p = new int[k];
		for(int i=0;i<k;i++) {
			v[i] = io.nextInt() - 1;
			p[i] = io.nextInt();
		}
		ans = new int[n];
		ArrayList<Integer> ans = null;
//		ArrayList<Integer> uso = new ArrayList<>();
//		for(int i=0;i<n;i++) {
//			uso.add(i);
//		}
//		Collections.shuffle(uso);
//		int max = Math.min(100, n);
//		for(int i=0;i<max;i++) {
			ans = solve(0);
//			if (ans != null) break;
//		}
		if (ans == null) {
			System.out.println("No");
		}else{
			io.println("Yes");
			for(int x: ans) {
				io.println(x);
			}
			io.flush();
		}
	}
	
	public ArrayList<Integer> solve(int s1) {
		Arrays.fill(ub, INF);
		Arrays.fill(lb, -INF);
		Arrays.fill(ans, 0);
		for(int i=0;i<k;i++) {
			ub[v[i]] = lb[v[i]] = p[i];
		}
		dfsUp(s1,-1);
		dfsDown(s1,-1);
		int s = -1;
		for(int i=0;i<n;i++) {
			if (ub[i] < lb[i]) {
				return null;
			}
			if (ub[i] == lb[i]) {
				s = i;
			}
		}
		ans[s] = lb[s];
		fail = false;
		dfsDown2(s, -1);
		if (fail) {
			return null;
		}
		for(int i=0;i<n;i++) {
			for(int j: tree[i]) {
				if (Math.abs(ans[i] - ans[j]) != 1) {
					return null;
				}
			}
		}
		ArrayList<Integer> al = new ArrayList<>();
		for(int i=0;i<n;i++) {
			al.add(ans[i]);
		}
		return al;
	}
	
	public int randInt(int min,int max) {
		return (int) (Math.random() * (max - min + 1)) + min;
	}
	
	public void dfsUp(int v,int p) {
		for(int c: tree[v]) {
			if (c == p) continue;
			dfsUp(c, v);
			ub[v] = Math.min(ub[v], ub[c] + 1);
			lb[v] = Math.max(lb[v], lb[c] - 1);
		}
	}
	
	public void dfsDown(int v,int p) {
		for(int c: tree[v]) {
			if (c == p) continue;
			ub[c] = Math.min(ub[c], ub[v] + 1);
			lb[c] = Math.max(lb[c], lb[v] - 1);
			dfsDown(c, v);
		}
	}
	
	public boolean fail = false;
	public void dfsDown2(int v,int p) {
		for(int c: tree[v]) {
			if (c == p) continue;
			boolean ok = false;
			for(int k=-1;k<=1;k+=2) {
				int next = ans[v] + k;
				if (lb[c] <= next && next <= ub[c]) {
					ans[c] = next;
					ok = true;
					break;
				}
			}
			if (!ok) {
				fail = true;
				return;
			}
			dfsDown2(c, v);
			if (fail) {
				return;
			}
		}
	}

}
class IO extends PrintWriter {
	private final InputStream in;
	private final byte[] buffer = new byte[1024];
	private int ptr = 0;
	private int buflen = 0;
	
	public IO() { this(System.in);}
	public IO(InputStream source) { super(System.out); this.in = source;}
	private boolean hasNextByte() {
		if (ptr < buflen) {
			return true;
		}else{
			ptr = 0;
			try {
				buflen = in.read(buffer);
			} catch (IOException e) {
				e.printStackTrace();
			}
			if (buflen <= 0) {
				return false;
			}
		}
		return true;
	}
	private int readByte() { if (hasNextByte()) return buffer[ptr++]; else return -1;}
	private static boolean isPrintableChar(int c) { return 33 <= c && c <= 126;}
	private static boolean isNewLine(int c) { return c == '\n' || c == '\r';}
	public boolean hasNext() { while(hasNextByte() && !isPrintableChar(buffer[ptr])) ptr++; return hasNextByte();}
	public boolean hasNextLine() { while(hasNextByte() && isNewLine(buffer[ptr])) ptr++; return hasNextByte();}
	public String next() {
		if (!hasNext()) {
			throw new NoSuchElementException();
		}
		StringBuilder sb = new StringBuilder();
		int b = readByte();
		while(isPrintableChar(b)) {
			sb.appendCodePoint(b);
			b = readByte();
		}
		return sb.toString();
	}
	public char[] nextCharArray(int len) {
		if (!hasNext()) {
			throw new NoSuchElementException();
		}
		char[] s = new char[len];
		int i = 0;
		int b = readByte();
		while(isPrintableChar(b)) {
			if (i == len) {
				throw new InputMismatchException();
			}
			s[i++] = (char) b;
			b = readByte();
		}
		return s;
	}
	public String nextLine() {
		if (!hasNextLine()) {
			throw new NoSuchElementException();
		}
		StringBuilder sb = new StringBuilder();
		int b = readByte();
		while(!isNewLine(b)) {
			sb.appendCodePoint(b);
			b = readByte();
		}
		return sb.toString();
	}
	public long nextLong() {
		if (!hasNext()) {
			throw new NoSuchElementException();
		}
		long n = 0;
		boolean minus = false;
		int b = readByte();
		if (b == '-') {
			minus = true;
			b = readByte();
		}
		if (b < '0' || '9' < b) {
			throw new NumberFormatException();
		}
		while(true){
			if ('0' <= b && b <= '9') {
				n *= 10;
				n += b - '0';
			}else if(b == -1 || !isPrintableChar(b)){
				return minus ? -n : n;
			}else{
				throw new NumberFormatException();
			}
			b = readByte();
		}
	}
	public int nextInt() {
		long nl = nextLong();
		if (nl < Integer.MIN_VALUE || nl > Integer.MAX_VALUE) {
			throw new NumberFormatException();
		}
		return (int) nl;
	}
	public char nextChar() {
		if (!hasNext()) {
			throw new NoSuchElementException();
		}
		return (char) readByte();
	}
	public double nextDouble() { return Double.parseDouble(next());}
	public int[] nextIntArray(int n) { int[] a = new int[n]; for(int i=0;i<n;i++) a[i] = nextInt(); return a;}
	public long[] nextLongArray(int n) { long[] a = new long[n]; for(int i=0;i<n;i++) a[i] = nextLong(); return a;}
	public double[] nextDoubleArray(int n) { double[] a = new double[n]; for(int i=0;i<n;i++) a[i] = nextDouble(); return a;}
	public void nextIntArrays(int[]... a) { for(int i=0;i<a[0].length;i++) for(int j=0;j<a.length;j++) a[j][i] = nextInt();}
	public int[][] nextIntMatrix(int n,int m) { int[][] a = new int[n][]; for(int i=0;i<n;i++) a[i] = nextIntArray(m); return a;}
	public char[][] nextCharMap(int n,int m) { char[][] a = new char[n][]; for(int i=0;i<n;i++) a[i] = nextCharArray(m); return a;}
	public void close() { super.close(); try {in.close();} catch (IOException e) {}}
}

