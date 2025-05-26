import java.io.*;
import java.util.*;
import java.math.*;
// import java.awt.Point;
 
public class Main {
    InputStream is;
    PrintWriter out;
    String INPUT = "";
 
    static int mod = 1_000_000_007;
    // int mod = 998244353;
    // long inf = Long.MAX_VALUE/2;
    int inf = Integer.MAX_VALUE/2;

    void solve(){
        int n = ni();
        long m = nl();
        int[] a = new int[n];
        for(int i = 0; i < n; i++){
            a[i] = ni();
        }
        int[] x = new int[100001];
        int[] y = new int[100001];
        for(int i = 0; i < n; i++){
            x[a[i]] += 1;
            y[a[i]] += 1;
        }
        double[][] ret = convolve(x,y);
        long ans = 0;
        long res = m;
        for (int i = 2*100000; i >= 1; i--) { 
            long val = Math.round(ret[0][i]);
            if(val==0) continue;
            if(res>=val){
                res -= val;
                ans += val*i;
            }
            else{
                ans += i*res;
                break;
            }
		}
        out.println(ans);
    }

    private static double[][] convolve(int[] a, int[] b) {
		// 次数上界がna(>0)である多項式A(x)の係数ベクトルをa[]、
		// 次数上界がnb(>0)である多項式B(x)の係数ベクトルをb[]とした時に、
		// A(x)とB(x)の乗算の結果である多項式C(x)の係数ベクトルを戻り値として返す。
		// C(x)の次数上界はna+nb-1。
		// a, bの配列は変更しない
		// 複素演算を用いたFFT。(再帰なし。アルゴリズムイントロダクション30.3参照)
		int na = a.length;
		int nb = b.length;
		int m = Math.max(na, nb);
		int n = Integer.highestOneBit(2 * m - 1) << 1;

		double[][] ca = new double[2][n];
		double[][] cb = new double[2][n];
		for (int i = 0; i < na; i++) {
			ca[0][i] = a[i];
		}
		for (int i = 0; i < nb; i++) {
			cb[0][i] = b[i];
		}

		ca = FFT(ca, false);
		cb = FFT(cb, false);

		for (int i = 0; i < n; i++) {
//			ca[i] = ca[i].multiply(cb[i]);
			double rtmp = ca[0][i] * cb[0][i] - ca[1][i] * cb[1][i];
			double itmp = ca[0][i] * cb[1][i] + ca[1][i] * cb[0][i];
			ca[0][i] = rtmp;
			ca[1][i] = itmp;
		}

//		Complex[] ret = Arrays.copyOf(FFT(ca, true), na + nb - 1);
		double[][] ret = FFT(ca, true);
		ret[0] = Arrays.copyOf(ret[0], na + nb - 1);
		ret[1] = Arrays.copyOf(ret[1], na + nb - 1);
		for (int i = 0; i < na + nb - 1; i++) {
			ret[0][i] /= n;
			ret[1][i] /= n;
		}

		return ret;
	}

	private static double[][] FFT(double[][] a, boolean inverse) {
		// 複素数の係数ベクトルaを持つ多項式を、
		// 1の原始n乗根のべき乗の点で評価(evaluation)する
		// inverseがtrueの時は、補間(interporation)になる。(1/nはしない)
		// aの内容は戻り値に変更される。
		int n = a[0].length;

//		Complex[] ret = new Complex[n];
		double[][] ret = a;
//		double[][] ret = new double[2][n];
		for (int k = 0; k < n; k++) {
			int rev = Integer.reverse(k);
			rev >>>= Integer.numberOfLeadingZeros(n) + 1;

			if (k < rev) {
				double tmp = ret[0][k];
				ret[0][k] = ret[0][rev];
				ret[0][rev] = tmp;
				tmp = ret[1][k];
				ret[1][k] = ret[1][rev];
				ret[1][rev] = tmp;
			}
//			ret[0][rev] = a[0][k];
//			ret[1][rev] = a[1][k];
		}

		for (int m = 2; m <= n; m *= 2) {
			double th = 2 * Math.PI / m;
			double[] wn = {Math.cos(th), Math.sin(th)};
			if (inverse) {
				wn[1] = -wn[1];
			}

			for (int k = 0; k < n; k += m) {
//				Complex w = new Complex(1, 0);
				double[] w = new double[2];
				w[0] = 1;

//				Complex t = new Complex(0, 0);
				double[] t = new double[2];
				for (int j = 0; j < m / 2; j++) {
//					Complex t = w.multiply(ret[k + j + m / 2]);
					double tmp0 = ret[0][k + j + m / 2];
					double tmp1 = ret[1][k + j + m / 2];
					t[0] = w[0] * tmp0 - w[1] * tmp1;
					t[1] = w[0] * tmp1 + w[1] * tmp0;
					ret[0][k + j + m / 2] = ret[0][k + j] - t[0];
					ret[1][k + j + m / 2] = ret[1][k + j] - t[1];
					ret[0][k + j] = ret[0][k + j] + t[0];
					ret[1][k + j] = ret[1][k + j] + t[1];

//					w = w.multiply(wn);
					double rtmp = w[0] * wn[0] - w[1] * wn[1];
					double itmp = w[0] * wn[1] + w[1] * wn[0];
					w[0] = rtmp;
					w[1] = itmp;
				}
			}
		}

		return ret;
	}

    void run() throws Exception
    {
        is = INPUT.isEmpty() ? System.in : new ByteArrayInputStream(INPUT.getBytes());
        out = new PrintWriter(System.out);
        long s = System.currentTimeMillis();
        solve();
        out.flush();
        if(!INPUT.isEmpty())tr(System.currentTimeMillis()-s+"ms");
    }
    
    public static void main(String[] args) throws Exception { new Main().run(); }
    
    private byte[] inbuf = new byte[1024];
    private int lenbuf = 0, ptrbuf = 0;
    
    private int readByte()
    {
        if(lenbuf == -1)throw new InputMismatchException();
        if(ptrbuf >= lenbuf){
            ptrbuf = 0;
            try { lenbuf = is.read(inbuf); } catch (IOException e) { throw new InputMismatchException(); }
            if(lenbuf <= 0)return -1;
        }
        return inbuf[ptrbuf++];
    }
    
    private boolean isSpaceChar(int c) { return !(c >= 33 && c <= 126); }
    private int skip() { int b; while((b = readByte()) != -1 && isSpaceChar(b)); return b; }
    
    private double nd() { return Double.parseDouble(ns()); }
    private char nc() { return (char)skip(); }
    
    private String ns()
    {
        int b = skip();
        StringBuilder sb = new StringBuilder();
        while(!(isSpaceChar(b) && b != ' ')){
            sb.appendCodePoint(b);
            b = readByte();
        }
        return sb.toString();
    }
    
    private char[] ns(int n)
    {
        char[] buf = new char[n];
        int b = skip(), p = 0;
        while(p < n && !(isSpaceChar(b))){
            buf[p++] = (char)b;
            b = readByte();
        }
        return n == p ? buf : Arrays.copyOf(buf, p);
    }
    
    private char[][] nm(int n, int m)
    {
        char[][] map = new char[n][];
        for(int i = 0;i < n;i++)map[i] = ns(m);
        return map;
    }
    
    private int[] na(int n)
    {
        int[] a = new int[n];
        for(int i = 0;i < n;i++)a[i] = ni();
        return a;
    }
    
    private int ni()
    {
        int num = 0, b;
        boolean minus = false;
        while((b = readByte()) != -1 && !((b >= '0' && b <= '9') || b == '-'));
        if(b == '-'){
            minus = true;
            b = readByte();
        }
        
        while(true){
            if(b >= '0' && b <= '9'){
                num = num * 10 + (b - '0');
            }else{
                return minus ? -num : num;
            }
            b = readByte();
        }
    }
    
    private long nl()
    {
        long num = 0;
        int b;
        boolean minus = false;
        while((b = readByte()) != -1 && !((b >= '0' && b <= '9') || b == '-'));
        if(b == '-'){
            minus = true;
            b = readByte();
        }
        
        while(true){
            if(b >= '0' && b <= '9'){
                num = num * 10 + (b - '0');
            }else{
                return minus ? -num : num;
            }
            b = readByte();
        }
    }
    
    private static void tr(Object... o) { System.out.println(Arrays.deepToString(o)); }
 
}
