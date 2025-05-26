import static java.lang.System.out;
import java.io.InputStream;
import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class Main {
    public static void main(String args[]) {
        MyInput in = new MyInput();
        int H = in.nextInt();
        int W = in.nextInt();
        int d = in.nextInt();
        String[][] map = new String[H][W];

        String[] clr = {"R","Y","G","B"};
        boolean[][][] used = new boolean[4][H][W];
        int tmp = 0;
        for(int i = 0; i < H; i++) {
            for(int k = 0; k < W; k++) {
                for(int l = 0 ; l < 4; l++) {
                    if(!used[l][i][k]) {
                        map[i][k] = clr[l];
                        used[l][i][k] = true;
                        tmp = l;
                    }
                }
                for(int l = 0 ; l <= d; l++) {
                    if(i+l < H && k+d-l < W) {
                        used[tmp][i+l][k+(d-l)] = true;
                    }
                    if(i-l >= 0 && k-(d-l) >= 0) {
                        used[tmp][i-l][k-(d-l)] = true;
                    }
                    if(i+l < H && k-(d-l) >= 0) {
                        used[tmp][i+l][k-(d-l)] = true;
                    }
                    if(i-l >= 0 && k+(d-l) < W) {
                        used[tmp][i-l][k+(d-l)] = true;
                    }
                }
            }
        }
        for(int i = 0; i < H; i++) {
            for(int k = 0; k < W; k++) {
                out.print(map[i][k]);
            }
            out.println();
        }

    }
}

class MyInput {
	private final InputStream in = System.in;
	private final byte[] buffer = new byte[1024];
	private int ptr= 0;
	private int buflen = 0;
	private boolean hasNextByte() {
		if(ptr < buflen) {
			return true;
		}else {
			ptr = 0;
			try {
				buflen = in.read(buffer);
			} catch(IOException e) {
				e.printStackTrace();
			}
			if(buflen <= 0) {
				return false;
			}
		}
		return true;
	}

	public int readByte() { 
		if (hasNextByte()) return buffer[ptr++];
		else return -1;
	}

	private static boolean isPrintableChar(int c) {
		return 33 <= c && c <=126;
	}

	private void skipUnprintable() {
		while(hasNextByte() && !isPrintableChar(buffer[ptr])) ptr++;
	}

	public boolean hasNext() {
		skipUnprintable();
		return hasNextByte();
	}

	public String next() {
		if(!hasNext()) throw new NoSuchElementException();
		StringBuilder sb = new StringBuilder();
		int b = readByte();
		while(isPrintableChar(b)) {
			sb.appendCodePoint(b);
			b = readByte();
		}
		return sb.toString();
	}

	public long nextLong() {
		if(!hasNext()) throw new NoSuchElementException();
		long n = 0;
		boolean minus = false;
		int b = readByte();
		if (b == '-') {
			minus = true;
			b = readByte();
		}
		if(b < '0' || '9' < b) {
			throw new NumberFormatException();
		}
		while(true) {
			if('0' <= b && b <= '9') {
				n *= 10;
				n += b - '0';
			}else if(b == -1 || !isPrintableChar(b)) {
				return minus ? -n : n;
			}else {
				throw new NumberFormatException();
			}
			b = readByte();
		}
	}

	public int nextInt() {
		long nl = nextLong();
		if(nl < Integer.MIN_VALUE || nl > Integer.MAX_VALUE) throw new NumberFormatException();
		return (int) nl;
	}

	public double nextDouble() {
		return Double.parseDouble(next());
	}
}

