import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.NoSuchElementException;
import java.util.PriorityQueue;
import java.util.concurrent.PriorityBlockingQueue;

public class Main {

    public static void main(String[] args) {
        FastScanner sc = new FastScanner();
        int n = sc.nextInt();
        int ma = sc.nextInt();
        int mb = sc.nextInt();
        int[][] r = new int[n][3];
        for (int i = 0; i < r.length; i++) {
            r[i][0] = sc.nextInt();
            r[i][1] = sc.nextInt();
            r[i][2] = sc.nextInt();
        }

        final int DSIZE = 440;
        final int DATA_MAX = 10000000;
        int[][] data = new int[DSIZE][DSIZE];
        for (int i = 0; i < DSIZE; i++) {
            for (int j = 0; j < DSIZE; j++) {
                data[i][j] = DATA_MAX;
            }
        }
        data[0][0] = 0;

        for (int i = 0; i < r.length; i++) {
            for (int j = DSIZE - 1; j >= 0; j--) {
                for (int j2 = DSIZE - 1; j2 >= 0; j2--) {
                    if(data[j][j2] != DATA_MAX){
                        data[j+r[i][0]][j2+r[i][1]] = Math.min(data[j][j2] + r[i][2], data[j+r[i][0]][j2+r[i][1]]);
                    }
                }
            }
        }

        int mma = 0;
        int mmb = 0;
        int ans = DATA_MAX;
        while(true){
            mma += ma;
            mmb += mb;
            if(mma >= DSIZE || mmb >= DSIZE) break;
            ans = Math.min(ans, data[mma][mmb]);
        }

        if(ans == DATA_MAX){
            System.out.println(-1);
        }else{
            System.out.println(ans);
        }
    }

    private static class FastScanner {
        private final InputStream in = System.in;
        private final byte[] buffer = new byte[1024];
        private int ptr = 0;
        private int buflen = 0;

        private boolean hasNextByte() {
            if (ptr < buflen) {
                return true;
            } else {
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

        private int readByte() {
            if (hasNextByte())
                return buffer[ptr++];
            else
                return -1;
        }

        private static boolean isPrintableChar(int c) {
            return 33 <= c && c <= 126;
        }

        public boolean hasNext() {
            while (hasNextByte() && !isPrintableChar(buffer[ptr]))
                ptr++;
            return hasNextByte();
        }

        public String next() {
            if (!hasNext())
                throw new NoSuchElementException();
            StringBuilder sb = new StringBuilder();
            int b = readByte();
            while (isPrintableChar(b)) {
                sb.appendCodePoint(b);
                b = readByte();
            }
            return sb.toString();
        }

        public long nextLong() {
            if (!hasNext())
                throw new NoSuchElementException();
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
            while (true) {
                if ('0' <= b && b <= '9') {
                    n *= 10;
                    n += b - '0';
                } else if (b == -1 || !isPrintableChar(b)) {
                    return minus ? -n : n;
                } else {
                    throw new NumberFormatException();
                }
                b = readByte();
            }
        }

        public int nextInt() {
            long nl = nextLong();
            if (nl < Integer.MIN_VALUE || nl > Integer.MAX_VALUE)
                throw new NumberFormatException();
            return (int) nl;
        }

        public double nextDouble() {
            return Double.parseDouble(next());
        }
    }

}
