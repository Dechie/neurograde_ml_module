import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.NoSuchElementException;
 
public class Main {
	static FastScanner sc = new FastScanner();
	static PrintStream out = System.out;
	static long sumX[];
	static long sumY[];
	static int bucketX[];
	static int bucketXidx[];
	static int bucketY[];
	static int bucketYidx[];
	static int bucketXSize, bucketYSize;
	static int bucketXLength, bucketYLength;
 
	public static void main(String args[]) {
		int W = ni();
		int H = ni();
		int p[] = na(W);
		int q[] = na(H);
		sumX = new long[W+1];
		sumY = new long[H+1];
		for (int x=0;x<W;x++) {
			sumX[x+1] = sumX[x] + p[x];
		}
		for (int y=0;y<H;y++) {
			sumY[y+1] = sumY[y] + q[y];
		}
		bucketXSize = (int) Math.floor(Math.sqrt(W));
		bucketYSize = (int) Math.floor(Math.sqrt(H));
		bucketXLength = W / bucketXSize+1;
		bucketYLength = H / bucketYSize+1;
		bucketX    = new int[bucketXLength];
		bucketXidx = new int[bucketXLength];
		bucketY    = new int[bucketYLength];
		bucketYidx = new int[bucketYLength];
		
		for (int i=0;i<W;i++) {
			int j = i/bucketXSize;
			if (bucketX[j] < p[i]) {
				bucketX[j] = p[i];
				bucketXidx[j] = i;
			}
		}

		for (int i=0;i<H;i++) {
			int j = i/bucketYSize;
			if (bucketY[j] < q[i]) {
				bucketY[j] = q[i];
				bucketYidx[j] = i;
			}
		}
 
		System.out.println(f(0, 0, W, H, p, q, 0, 0, 0, 0));
	}

	private static int[] getMaxX(int a, int b, int p[]) {
		int s1 = (a + (bucketXSize - 1)) / bucketXSize;
		int s2 = b / bucketXSize;
		int max = 0;
		int idx = 0;
		for (int i=s1;i<s2;i++) {
			if (max < bucketX[i]) {
				max = bucketX[i];
				idx = bucketXidx[i];
			}
		}

		for (int i=a;i<s1*bucketXSize;i++) {
			if (max < p[i]) {
				max = p[i];
				idx = i;
			}
		}

		for (int i=s2*bucketXSize;i<b;i++) {
			if (max < p[i]) {
				max = p[i];
				idx = i;
			}
		}

		return new int[]{max, idx};
	}

	private static int[] getMaxY(int a, int b, int p[]) {
		int s1 = (a + (bucketYSize - 1)) / bucketYSize;
		int s2 = b / bucketYSize;
		int max = 0;
		int idx = 0;
		for (int i=s1;i<s2;i++) {
			if (max < bucketY[i]) {
				max = bucketY[i];
				idx = bucketYidx[i];
			}
		}

		for (int i=a;i<s1*bucketYSize;i++) {
			if (max < p[i]) {
				max = p[i];
				idx = i;
			}
		}

		for (int i=s2*bucketYSize;i<b;i++) {
			if (max < p[i]) {
				max = p[i];
				idx = i;
			}
		}

		return new int[]{max, idx};
	}
	
	private static long f(int x0, int y0, int x1, int y1, int p[], int q[], int maxX, int maxY, int maxXidx, int maxYidx) {
		if (x0 == x1) {
			long ret = 0;
			ret = sumY[y1] - sumY[y0];
			return ret;
		} else if (y0 == y1) {
			long ret = 0;
			ret = sumX[x1] - sumX[x0];
			return ret;
		}
 
		if (maxX == 0) {
			int r[] = getMaxX(x0, x1, p);
			maxX = r[0];
			maxXidx = r[1];
		}

		if (maxY == 0) {
			int r[] = getMaxY(y0, y1, q);
			maxY = r[0];
			maxYidx = r[1];
		}
 
		if (maxX > maxY) {
			long ret =
					f(x0, y0, maxXidx, y1, p, q, 0, maxY, 0, maxYidx) + 
					f(maxXidx+1, y0, x1, y1, p, q, 0, maxY, 0, maxYidx) + 
					maxX;
			return ret;
		} else {
			long ret =
					f(x0, y0, x1, maxYidx, p, q, maxX, 0, maxXidx, 0) + 
					f(x0, maxYidx+1, x1, y1, p, q, maxX, 0, maxXidx, 0) + 
					maxY;
			return ret;
		}
	}
 
	private static String next() {
		return sc.next();
	}
 
	private static int ni() {
		return sc.nextInt();
	}
 
	private static long nl() {
		return sc.nextLong();
	}
 
	private static int[] na(int n) {
		int[] a = new int[n];
		for(int i = 0;i < n;i++) a[i] = ni();
		return a;
	}
 
	private static long[] nal(int n) {
		long[] a = new long[n];
		for(int i = 0;i < n;i++) a[i] = ni();
		return a;
	}
 
    private static double nd() {
        return Double.parseDouble( next() );
    }
}
 
class FastScanner {
    private final InputStream in = System.in;
    private final byte[] buffer = new byte[1024];
    private int ptr = 0;
    private int buflen = 0;
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
    private void skipUnprintable() { while(hasNextByte() && !isPrintableChar(buffer[ptr])) ptr++;}
    public boolean hasNext() { skipUnprintable(); return hasNextByte();}
    public String next() {
        if (!hasNext()) throw new NoSuchElementException();
        StringBuilder sb = new StringBuilder();
        int b = readByte();
        while(isPrintableChar(b)) {
            sb.appendCodePoint(b);
            b = readByte();
        }
        return sb.toString();
    }
    public int nextInt() {
        if (!hasNext()) throw new NoSuchElementException();
        int n = 0;
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
    public long nextLong() {
        if (!hasNext()) throw new NoSuchElementException();
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
}