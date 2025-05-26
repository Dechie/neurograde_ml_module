import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.stream.IntStream;

public class Main {

    private void solve(FastScanner sc) {
        final int W = 0;
        final int H = 1;
        int w = sc.nextInt();
        int h = sc.nextInt();
        List<Pair> c = new ArrayList<>(w + h);
        Map<Integer, TreeSet<Integer>> pPos = new HashMap();
        Map<Integer, TreeSet<Integer>> qPos = new HashMap();
        for (int i = 0; i < w; i++) {
            int ni = sc.nextInt();
            if (!pPos.containsKey(ni)) {
                pPos.put(ni, new TreeSet<>());
            }
            pPos.get(ni).add(i);
            c.add(new Pair(ni, W));
        }
        for (int i = 0; i < h; i++) {
            int ni = sc.nextInt();
            if (!qPos.containsKey(ni)) {
                qPos.put(ni, new TreeSet<>());
            }
            qPos.get(ni).add(i);
            c.add(new Pair(ni, H));
        }
        long ans = 0;
        Collections.sort(c, (o1, o2) -> Integer.compare(o1.fst, o2.fst));
        UnionFind uf = new UnionFind(GETA*GETA);
        for (Pair pair : c) {
            int cost = pair.fst;
            int type = pair.snd;
            if (type == W) {
                for (int j = 0; j <= h; j++) {
                    for (int i : pPos.get(cost)) {
                        int x = toTuple(i, j);
                        int y = toTuple(i + 1, j);
                        if (!uf.same(x, y)) {
                            uf.unite(x, y);
                            ans += (long) cost;
                        }
                    }
                }
            } else {
                for (int i = 0; i <= w; i++) {
                    for (int j : qPos.get(cost)) {
                        int x = toTuple(i, j);
                        int y = toTuple(i, j + 1);
                        if (!uf.same(x, y)) {
                            uf.unite(x, y);
                            ans += (long) cost;
                        }
                    }
                }
            }
        }
        System.out.println(ans);
    }

    /*
     * template
     */

    static class Pair {

        public int fst;
        public int snd;

        public Pair(int fst, int snd) {
            this.fst = fst;
            this.snd = snd;
        }

        public Pair(List<Integer> list) {
            this.fst = list.get(0);
            this.snd = list.get(1);
        }

        public Pair(int[] array) {
            this.fst = array[0];
            this.snd = array[1];
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Pair)) return false;

            Pair pair = (Pair) o;

            return fst == pair.fst && snd == pair.snd;

        }

        @Override
        public int hashCode() {
            int result = fst;
            result = 31 * result + snd;
            return result;
        }

        @Override
        public String toString() {
            return "(" + fst + ", " + snd + ")";
        }

        public void swap() {
            int swap = fst;
            fst = snd;
            snd = swap;
        }
    }

    public static final int GETA = 10001;

    public static int toTuple(int fst, int snd) {
        return fst * GETA + snd;
    }

    public static int getFst(int t) {
        return t / GETA;
    }

    public static int getSnd(int t) {
        return t % GETA;
    }

    public static void main(String[] args) {
        FastScanner sc = new FastScanner();
        new Main().solve(sc);
    }

    static class FastScanner {
        private final InputStream in = System.in;
        private final byte[] buffer = new byte[1024];
        private int ptr = 0;
        private int bufLen = 0;

        private boolean hasNextByte() {
            if (ptr < bufLen) {
                return true;
            } else {
                ptr = 0;
                try {
                    bufLen = in.read(buffer);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                if (bufLen <= 0) {
                    return false;
                }
            }
            return true;
        }

        private int readByte() {
            if (hasNextByte()) return buffer[ptr++];
            else return -1;
        }

        private static boolean isPrintableChar(int c) {
            return 33 <= c && c <= 126;
        }

        boolean hasNext() {
            while (hasNextByte() && !isPrintableChar(buffer[ptr])) ptr++;
            return hasNextByte();
        }

        String next() {
            if (!hasNext()) throw new NoSuchElementException();
            StringBuilder sb = new StringBuilder();
            int b = readByte();
            while (isPrintableChar(b)) {
                sb.appendCodePoint(b);
                b = readByte();
            }
            return sb.toString();
        }

        long nextLong() {
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

        int nextInt() {
            long nl = nextLong();
            if (nl < Integer.MIN_VALUE || nl > Integer.MAX_VALUE) throw new NumberFormatException();
            return (int) nl;
        }

        double nextDouble() {
            return Double.parseDouble(next());
        }
    }

    static class UnionFind {

        private int[] par;
        private int[] rank;

        public UnionFind(int n) {
            par = IntStream.range(0, n).toArray();
            rank = new int[n];
        }

        public int find(int x) {
            return par[x] == x ? x : (par[x] = find(par[x]));
        }

        public void unite(int x, int y) {
            x = find(x);
            y = find(y);
            if (x == y) return;
            if (rank[x] < rank[y]) {
                par[x] = y;
            } else {
                par[y] = x;
                if (rank[x] == rank[y]) rank[x]++;
            }
        }

        public boolean same(int x, int y) {
            //noinspection SuspiciousNameCombination
            return find(x) == find(y);
        }
    }
}