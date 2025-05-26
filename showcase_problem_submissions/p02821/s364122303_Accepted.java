
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.math.BigInteger;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

/**
 * https://kenkoooo.com/atcoder/#/contest/show/5907bacd-a3b5-46d5-9f60-e3fc33f0341a
 */
public class Main {

    public Main() throws IOException {
    }

    public static void main(String[] args) throws IOException {
        new Main().solve();
    }

    //    private final InputStream in = Files.newInputStream(Paths.get("C:\\Users\\ryo.suzuki\\Downloads\\02.txt"));
    private final InputStream in = System.in;
    private final PrintWriter out = new PrintWriter(System.out);
    private final byte[] buffer = new byte[1024];
    private int ptr = 0;
    private int buflen = 0;

    private void solve() throws IOException {
        try {
//            solveA();
//            solveB();
//            solveC();
//            solveD();
//            solveE();
            solveF();
        } finally {
            if (in != null) {
                in.close();
            }
            if (out != null) {
                out.flush();
                out.close();
            }
        }

    }

    // mods
    long MOD = (long) Math.pow(10, 9) + 7;

    /**
     * https://atcoder.jp/contests/abc135/tasks/abc135_a
     * A - Harmony
     */
    private void solveA() {
        int a = nextInt();
        int b = nextInt();
        {
//            for (int i = 0; i <= pow(10, 9); i++) {
//                if (Math.abs(a - i) == Math.abs(b - i)) {
//                    out.println(i);
//                    return;
//                }
//                out.println("IMPOSSIBLE");
//            }
        }
        {
            if (Math.abs(a) % 2 == Math.abs(b) % 2)
                out.println((a + b) / 2);
            else
                out.println("IMPOSSIBLE");
        }
        {
//        if (Math.abs(a - b) == Math.abs(b - a) && Math.abs(b - a) % 2 == 0) {
//            out.println((a + b) / 2);
//        } else {
//            out.println("IMPOSSIBLE");
//        }
        }
    }

    /**
     * https://atcoder.jp/contests/abc072/tasks/arc082_a
     * C - Together
     */
    private void solveB() {
        int n = nextInt();
        int[] a = IntStream.range(0, n).map(i -> nextInt()).toArray();
        Map<Integer, Integer> memo = new HashMap<Integer, Integer>();
        for (int item : a) {
            memo.put(item, memo.getOrDefault(item, 0) + 1);
            memo.put(item - 1, memo.getOrDefault(item - 1, 0) + 1);
            memo.put(item + 1, memo.getOrDefault(item + 1, 0) + 1);
        }
        OptionalInt max = memo.values().stream().mapToInt(i -> i).max();
        out.println(max.getAsInt());
    }

    /**
     * https://atcoder.jp/contests/cf16-final/tasks/codefestival_2016_final_b
     * B - Exactly N points
     */
    private void solveC() {
        int n = nextInt();
        int sum = 0;
        int cnt = 1;
        Set<Integer> set = new HashSet<Integer>();
        while (sum <= n) {
            sum += cnt;
            set.add(cnt);
            cnt++;
        }
        int wk = sum - n;
        set.remove(wk);
        for (int item : set) {
            out.println(item);
        }
    }

    /**
     * https://atcoder.jp/contests/abc146/tasks/abc146_d
     * D - Coloring Edges on Tree
     */
    private void solveD() {
        int n = nextInt();
        int[][] ab = IntStream.range(0, n - 1).mapToObj(i -> new int[]{nextInt() - 1, nextInt() - 1}).toArray(int[][]::new);
        SimpleGraph graph = new SimpleGraph(ab);
        int[] color = new int[n];
        int[] colorOut = new int[n - 1];
        Queue<Integer> queue = new ArrayDeque<Integer>();
        queue.add(0);
        while (!queue.isEmpty()) {
            int cnt = queue.size();
            for (int i = 0; i < cnt; i++) {
                int idx = queue.poll();
                SimpleGraphNode nodes = graph.getNode(idx);
                int c = 1;
                for (Map.Entry<Integer, Integer> child : nodes.childs.entrySet()) {
                    if (color[child.getKey()] != 0)
                        continue;
                    if (color[idx] == c)
                        c++;
                    color[child.getKey()] = c;
                    colorOut[child.getValue()] = c;
                    c++;
                    queue.add(child.getKey());
                }
            }
        }
        out.println(Arrays.stream(colorOut).distinct().count());
        for (int item : colorOut) {
            out.println(item);
        }
    }

    /**
     * https://atcoder.jp/contests/arc093/tasks/arc093_b
     * D - Grid Components
     */
    private void solveE() {
        int a = nextInt() - 1;
        int b = nextInt() - 1;
        int k = 50;
        char[][] res = new char[2 * k][2 * k];
        /*
        10,000のGridを作成
        これで、a=1,b=1が達成
         */
        for (int i = 0; i < 2 * k; i++) {
            if (i < k)
                Arrays.fill(res[i], '#');
            else
                Arrays.fill(res[i], '.');
        }
        /*
        a-1,b-1分を相互につぶす
         */
        int cnt = 0;
        for (int i = 0; i < k; i += 2) {
            for (int j = 0; j < 2 * k; j += 2) {
                if (cnt == a)
                    break;
                res[i][j] = '.';
                cnt++;
            }
        }
        cnt = 0;
        for (int i = k + 1; i < 2 * k; i += 2) {
            for (int j = 0; j < 2 * k; j += 2) {
                if (cnt == b)
                    break;
                res[i][j] = '#';
                cnt++;
            }
        }
        out.println((2 * k) + " " + (2 * k));
        for (char[] item : res) {
            StringBuilder builder = new StringBuilder();
            for (char c : item) {
                builder.append(c);
            }
            out.println(builder.toString());
        }
    }


    /**
     * https://atcoder.jp/contests/abc149/tasks/abc149_e
     * E - Handshake
     * <p>
     * https://tt-conpetitive.hatenablog.com/entry/2020/01/03/221433
     * https://scrapbox.io/ganariya/AtCoderBeginnerContest149_E%E5%95%8F%E9%A1%8C500%E7%82%B9_%E3%80%8CHandshake%E3%80%8D
     */
    private void solveF() {
        int n = nextInt();
        long m = nextLong();
        long[] a = LongStream.range(0, n).map(i -> nextLong()).sorted().toArray();
        /**
         * 		10	14	19	33	34
         * 10	20	24	29	43	44
         * 14	24	28	33	47	48
         * 19	29	33	38	52	53
         * 33	43	47	52	66	67
         * 34	44	48	53	67	68
         * =67+67+68
         * <p>
         * 		1	3	3	5	5	21	24	34	110
         * 1	2	4	4	6	6	22	25	35	111
         * 3	4	6	6	8	8	24	27	37	113
         * 3	4	6	6	8	8	24	27	37	113
         * 5	6	8	8	10	10	26	29	39	115
         * 5	6	8	8	10	10	26	29	39	115
         * 21	22	24	24	26	26	42	45	55	131
         * 24	25	27	27	29	29	45	48	58	134
         * 34	35	37	37	39	39	55	58	68	144
         * 110	111	113	113	115	115	131	134	144	220
         * = 113以上の数の合計 - mを超えた分の113
         */
        long left = -1;
        long right = a[a.length - 1] + a[a.length - 1] + 1;
		/*
		-1 から wk[max]*2+1　の間で、wk[i]+wk[i]の組数がmより小さい時の最大値を探す
		m以上の時の最小値(right)のばあい、mに満たない可能性がある。
		そのため、targetは組合せのショートを考慮してleftを採用
		これにより組合せ個数がオーバーの可能性があるがそこは後で合わせる。
		 */
        while (right - left > 1) {
            long mid = left + (right - left) / 2;
            /*
            組合せがm個以上いるmidということは、midがちょっと小さい可能性がある
            そのため、leftをずらす。逆ならrightをずらす
             */
            if (!chkHandShakeNum(a, mid, m))
                right = mid;
            else
                left = mid;
        }

        //この値以上となる組み合わせを合計する
        long target = right;
        long[] rui = new long[n + 1];
        for (int i = 0; i < n; i++)
            rui[i + 1] = rui[i] + a[i];

        long res = 0;

        /*
        targetが目指す点数
         */
        for (int i = 0; i < n; i++) {
        	/*
        	targetからa[i]を引いた残りがどこのindexにいるのかの確認
        	a[i]と組み合わせるとtarget以上となるa[i]の位置
        	 */
            int pos = lowerBound(a, target - a[i]);
            long cnt = n - pos;//累積和のためのポジション計算
            //a[i]と組み合わせるとtarget以上となる相手がいない
            if (cnt == 0)
                continue;
            //相手の数がcnt・a[i]×cntがベース。+相手の累積
            res += cnt * a[i] + (rui[n] - rui[pos]);//累積
            m -= cnt;//帳尻合わせのためにmの残数を確認
        }
        /*
        mが-まで行く可能性がある（重複選択）ため、-分targetを引いておく
        ここの段階で最初のにぶたんじにrightを採用すると、mに足りていない。
        かといって、*mを実施すると本来なら113なのに114を採用することになる。
         */
        res += m * left;
        out.println(res);
    }

    /**
     * val以上の個数の合計がM個以上かどうか
     * m個以上であれば一旦採用
     */
    private boolean chkHandShakeNum(long[] a, long val, long m) {
        long cnt = 0;
        /*
        left=a[i]を採用した時、合計をvalとするためには、
        rightのindexを何以上にするべきか？
        ↓
        a[i]を組み合わせてvalとするとしたら、何通りになるのか？
         */
        for (int i = 0; i < a.length; i++) {
			/*
			val-nums[i]の残りの値がどこのindexにいるのか確認する。
			これがmを超えるのならnums[i]を採用した場合には相手がいないことになる。
			 */
            long pos = lowerBound(a, val - a[i]);
            cnt += (a.length - pos);
        }
        return cnt >= m;
    }

    private int lowerBound(long[] a, long n) {
        int left = -1;
        int right = a.length;
        while (right - left > 1) {
            int mid = left + (right - left) / 2;
            if (a[mid] >= n)
                right = mid;
            else
                left = mid;
        }
        return right;
    }

    static final long CONST_INF = Long.MAX_VALUE / 4;


    private static class VectorCalculation {
        /**
         * 点Pと線(AB)の距離
         *
         * @param p
         * @param a
         * @param b
         * @return
         */
        private double distanceDotAndLine(DoublePair p, DoublePair a, DoublePair b) {
            DoublePair ab = new DoublePair(b.x - a.x, b.y - a.y);
            DoublePair ap = new DoublePair(p.x - a.x, p.y - a.y);

            // ベクトルAB、APの外積の絶対値が平行四辺形Dの面積になる
            double d = Math.abs(crossVector(ab, ap));

            double l = distanceVertex(a, b); // AB間の距離

            double h = d / l;
            return h;
        }

        /**
         * 2点間距離
         *
         * @param p1
         * @param p2
         * @return
         */
        private double distanceVertex(DoublePair p1, DoublePair p2) {
            double dx = p1.x - p2.x;
            double dy = p1.y - p2.y;
            return Math.sqrt(dx * dx + dy * dy);
        }

        /**
         * ベクトル外積
         *
         * @param vl
         * @param vr
         * @return
         */
        private double crossVector(DoublePair vl, DoublePair vr) {
            return vl.x * vr.y - vl.y * vr.x;
        }
    }

    private static class DoublePair implements Comparable<DoublePair> {
        double x;
        double y;

        public DoublePair(double k, double v) {
            this.x = k;
            this.y = v;
        }

        public int compareTo(DoublePair pair) {
            return Double.compare(this.y, pair.y);
        }
    }

    private static class SimplePair implements Comparable<SimplePair> {
        long k;
        long v;

        public SimplePair(int k, int v) {
            this.k = k;
            this.v = v;
        }

        public SimplePair(long k, long v) {
            this.k = k;
            this.v = v;
        }

        public int compareTo(SimplePair pair) {
            return Long.compare(this.v, pair.v);
        }
    }


    private static class SimpleGraphNode {
        private int n;
        private Map<Integer, Integer> childs;

        public SimpleGraphNode(int n) {
            this.n = n;
            childs = new HashMap<Integer, Integer>();
        }

        public void addChild(int child, int d) {
            this.childs.put(child, d);
        }
    }

    private static class SimpleGraph {

        public Map<Integer, SimpleGraphNode> graph;

        public SimpleGraph(int[][] list) {
            graph = new HashMap<Integer, SimpleGraphNode>();
            for (int i = 0; i < list.length; i++) {
                int a = list[i][0];
                int b = list[i][1];
                graph.put(a, graph.getOrDefault(a, new SimpleGraphNode(a)));
                graph.get(a).addChild(b, i);
            }

        }

        public SimpleGraphNode getNode(int i) {
            return graph.getOrDefault(i, new SimpleGraphNode(-1));
        }

    }

    private static class GraphNode {
        private long n;
        private Map<Long, SimplePair> childs;

        public GraphNode(long n) {
            this.n = n;
            childs = new TreeMap<Long, SimplePair>();
        }

        /**
         * not update
         *
         * @param child
         * @param d
         */
        public void addChild(long child, long d) {
            this.childs.put(child, this.childs.getOrDefault(child, new SimplePair(child, d)));
        }

        public SimplePair getChild(long child) {
            return this.childs.getOrDefault(child, new SimplePair(child, CONST_INF));
        }
    }

    private static class Graph {

        private Map<Long, GraphNode> graph;
        int n;

        /**
         * @param n    頂点数
         * @param list
         */
        public Graph(int n, int[][] list) {
            this.n = n;
            graph = new HashMap<Long, GraphNode>();
            /*
            隣接行列作成
            0-indexed
             */
            Arrays.stream(list).forEach(l -> {
                long a = l[0] - 1;
                long b = l[1] - 1;
                long d = l[2];
                graph.put(a, graph.getOrDefault(a, new GraphNode(a)));
                graph.get(a).addChild(b, d);
                graph.put(b, graph.getOrDefault(b, new GraphNode(b)));
                graph.get(b).addChild(a, d);
            });
            /*
            たどり着けない場所はCONST_INFで埋める
            自分へはCOST=0でたどり着ける
             */
            for (long i = 0; i < n; i++) {
                graph.put(i, graph.getOrDefault(i, new GraphNode(i)));
                GraphNode node = graph.get(i);
                for (int j = 0; j < n; j++) {
                    if (i == j)
                        node.addChild(j, 0L);
                    else
                        node.addChild(j, CONST_INF);
                }
            }
        }

        /**
         * ワーシャルフロイド
         */
        public void warshallFloyd() {
            for (long k = 0; k < n; k++) {
                for (long i = 0; i < n; i++) {
                    for (long j = 0; j < n; j++) {
                        long min = Math.min(graph.get(i).getChild(j).v, graph.get(i).getChild(k).v + graph.get(k).getChild(j).v);
                        graph.get(i).getChild(j).v = min;
                    }
                }
            }
        }

        public void partialUpdateAndWarshallFloyd(long x, long y, long w) {
            if (graph.get(x).getChild(y).v > w)
                graph.get(x).getChild(y).v = graph.get(y).getChild(x).v = w;
            long[] l = {x, y};
            for (long k : l) {
                for (long i = 0; i < n; i++) {
                    for (long j = 0; j < n; j++) {
                        if (graph.get(i).getChild(j).v > graph.get(i).getChild(k).v + graph.get(k).getChild(j).v) {
                            graph.get(i).getChild(j).v = graph.get(i).getChild(k).v + graph.get(k).getChild(j).v;
                        }
                    }
                }
            }
        }

        /**
         * 2点間の距離
         *
         * @param from
         * @param to
         * @return
         */
        public long getDistance(long from, long to) {
            return graph.get(from).getChild(to).v;
        }

    }

    public long calcSimpleCombination(long n, long m, long CONST_MOD) {
        long mole = 1;
        for (long i = 1; i <= n + m; i++) {
            mole *= i;
            mole %= CONST_MOD;
        }
        long deno = 1;
        for (long i = 1; i <= n; i++) {
            deno *= i;
            deno %= CONST_MOD;
        }
        for (int i = 1; i <= m; i++) {
            deno *= i;
            deno %= CONST_MOD;
        }
        deno = modInverse(deno, CONST_MOD);
        return (mole * deno) % CONST_MOD;
    }

    long abs(double x) {
        return (long) Math.abs(x);
    }

    long round(double x) {
        return Math.round(x);
    }

    long floor(double x) {
        return (long) Math.floor(x);
    }

    long ceil(double x) {
        return (long) Math.ceil(x);
    }

    double sqrt(double x) {
        return Math.sqrt(x);
    }

    double pow(double x, double y) {
        return Math.pow(x, y);
    }

    long pow(long x, long y) {
        return (long) Math.pow(x, y);
    }

    int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    long gcd(long a, long b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    long lcm(long a, long b) {
        return a * b / gcd(a, b);
    }

    public long lcmMod(long... ab) {

        Map<Long, Long> ps = new HashMap<Long, Long>();

        for (int abl = 0; abl < ab.length; abl++) {
            long target = ab[abl];
            long l = target;
            for (long p = 2; p * p <= l; p++) {
                long cnt = 0;
                while (target > 1 && target % p == 0) {
                    cnt++;
                    target /= p;
                }
                if (cnt == 0)
                    continue;
                ps.put(p, Long.max(cnt, ps.getOrDefault(p, 0L)));
            }
            if (target > 1)
                ps.put(target, Long.max(1, ps.getOrDefault(target, 0L)));
        }
        long res = 1L;
        for (Map.Entry<Long, Long> e : ps.entrySet()) {
            res *= powMod(e.getKey(), e.getValue());
            res %= MOD;
        }
        return res;
    }

    int upperToInt(char a) {
        return a - 'A';
    }

    int lowerToInt(char a) {
        return a - 'a';
    }

    int numToInt(char a) {
        return a - '0';
    }

    int charToInt(char a) {
        return a >= 'a' ? lowerToInt(a) : a >= 'A' ? upperToInt(a) : numToInt(a);
    }

    char intToUpper(int a) {
        return (char) (a + 'A');
    }

    char intToLower(int a) {
        return (char) (a + 'a');
    }

    char intToNum(int a) {
        return (char) (a + '0');
    }

    void reverse(String array[]) {
        String reversed[] = new String[array.length];
        for (int i = 0; i < array.length; i++) {
            reversed[array.length - i - 1] = array[i];
        }
        for (int i = 0; i < array.length; i++) {
            array[i] = reversed[i];
        }
    }

    void reverse(int array[]) {
        int reversed[] = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            reversed[array.length - i - 1] = array[i];
        }
        for (int i = 0; i < array.length; i++) {
            array[i] = reversed[i];
        }
    }

    void reverse(long array[]) {
        long reversed[] = new long[array.length];
        for (int i = 0; i < array.length; i++) {
            reversed[array.length - i - 1] = array[i];
        }
        for (int i = 0; i < array.length; i++) {
            array[i] = reversed[i];
        }
    }

    void reverse(double array[]) {
        double reversed[] = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            reversed[array.length - i - 1] = array[i];
        }
        for (int i = 0; i < array.length; i++) {
            array[i] = reversed[i];
        }
    }

    void reverse(boolean array[]) {
        boolean reversed[] = new boolean[array.length];
        for (int i = 0; i < array.length; i++) {
            reversed[array.length - i - 1] = array[i];
        }
        for (int i = 0; i < array.length; i++) {
            array[i] = reversed[i];
        }
    }

    void fill(int array[][], int x) {
        for (int a[] : array) {
            Arrays.fill(a, x);
        }
    }

    void fill(long array[][], long x) {
        for (long a[] : array) {
            Arrays.fill(a, x);
        }
    }

    void fill(double array[][], double x) {
        for (double a[] : array) {
            Arrays.fill(a, x);
        }
    }

    void fill(boolean array[][], boolean x) {
        for (boolean a[] : array) {
            Arrays.fill(a, x);
        }
    }

    void fill(int array[][][], int x) {
        for (int a[][] : array) {
            fill(a, x);
        }
    }

    void fill(long array[][][], long x) {
        for (long a[][] : array) {
            fill(a, x);
        }
    }

    void fill(double array[][][], double x) {
        for (double a[][] : array) {
            fill(a, x);
        }
    }

    void fill(boolean array[][][], boolean x) {
        for (boolean a[][] : array) {
            fill(a, x);
        }
    }

    long L_INF = (long) 1e18 + 7;

    boolean isINF(long a) {
        return abs(a) > L_INF / 1000;
    }

    boolean isPlusINF(long a) {
        return a > 0 && isINF(a);
    }

    boolean isMinusINF(long a) {
        return isPlusINF(-a);
    }

    int I_INF = (int) 1e9 + 7;

    boolean isINF(int a) {
        return abs(a) > I_INF / 1000;
    }

    boolean isPlusINF(int a) {
        return a > 0 && isINF(a);
    }

    boolean isMinusINF(int a) {
        return isPlusINF(-a);
    }

    /**
     * 指定したx以下の素数をリストアップする
     */
    public long[] getPrimesUnderX(long x) {
        return LongStream.rangeClosed(2, x)
                .filter(i -> LongStream.rangeClosed(2, (long) Math.sqrt(i)).allMatch(j -> i % j != 0)).toArray();
    }


    public long mod(long i) {
        return i % MOD + ((i % MOD) < 0 ? MOD : 0);
    }

    long powMod(long x, long y) {
        if (y == 0) {
            return 1;
        } else {
            long tmp = powMod(x, y / 2);
            return mod(mod(tmp * tmp) * (y % 2 == 0 ? 1 : x));
        }
    }

    long inv(long x) {
        return powMod(x, MOD - 2);
    }

    int MAX_FACT = 5_000_100;
    long fact[];
    long invFact[];

    /**
     * Combination簡易版
     * 5 C 2
     * 異なる n個のものから r個を選ぶ組み合わせの総数 nCr を求めます。
     * 5!(5*4*3*2*1)
     * /
     * 2!(2*1) *  (5-2)!(3*2*1)
     *
     * @param n
     * @param r
     * @return
     */
    private long getComb(int n, int r) {
        long tmp = 1;
        for (int i = 1; i <= r; i++) {
            tmp *= n - i + 1;
            tmp = mod(tmp);
            tmp *= inv(i);
            tmp = mod(tmp);
        }
        return tmp;
    }

    /**
     * 階乗計算の事前累積和
     * [1, 1, 2, 3, 4, 5, … FACTORIAL_NUM]
     * mod済
     */
    void prepareFact() {
        fact = new long[MAX_FACT];
        Arrays.fill(fact, 0);
        invFact = new long[MAX_FACT];
        Arrays.fill(invFact, 0);
        fact[0] = 1;
        int maxIndex = (int) min(MAX_FACT, (int) MOD);
        for (int i = 1; i < maxIndex; i++) {
            fact[i] = mod(fact[i - 1] * i);
        }
        invFact[maxIndex - 1] = inv(fact[maxIndex - 1]);
        for (int i = maxIndex - 1; i > 0; i--) {
            invFact[i - 1] = mod(invFact[i] * i);
        }
    }

    /**
     * 順列
     * nPk -> n! / (n-k)!
     *
     * @param n
     * @param r
     * @return
     */
    long permutation(int n, int r) {
        if (n < 0 || r < 0 || n < r) {
            return 0;
        }
        return mod(fact[n] * invFact[n - r]);
    }

    /**
     * 組み合わせ
     * nCk -> n! / k!・(n-k)!
     *
     * @param n
     * @param r
     * @return
     */
    long combination(int n, int r) {
        if (n < 0 || r < 0 || n < r) {
            return 0;
        }
        return mod(permutation(n, r) * invFact[r]);
    }

    /**
     * 重複組合せ nHr （同次積）
     * nHr = (n+r-1)Cr
     * 異なるn個のものから重複を許してr個取る組合せの総数
     * 例：
     * リンゴ，ミカン，牛肉の3種類の果物があります．これらの中から6個の食材を買って帰ります．
     * このとき，何通りの買い方がありますか？ただし，含まれない食材があってもよいものとします
     *
     * @param n
     * @param r
     * @return
     */
    long homogeneousProduct(int n, int r) {
        return combination((n - 1) + r, r);
    }

    /**
     * 多項係数
     * 文字aをp個,bをq個,cをr個, dをs個 あわせてn個を1列に並べるときの並べ方
     * n! / p!・q!・r!・s!
     *
     * @param n
     * @param strNum
     * @param mod
     * @return
     */

    /**
     * フェルマーの小定理を用いて逆元を求める。
     * ある数xのmod p（pは素数）の逆数x'はx' = x^(p-2)
     * 繰り返し二乗法を用いて計算する。
     * http://satanic0258.hatenablog.com/entry/2016/04/29/004730
     * {@link BigInteger#modInverse(BigInteger)}とどちらが速いか？
     *
     * @param x
     * @return
     */
    private long modInverse(long x, long mod) {
        long res = 1L;
        long k = mod - 2L;
        long y = x;
        while (k != 0) {
            if (k % 2 != 0) {
                res = (res * y) % mod;
            }
            y = (y * y) % mod;
            k /= 2;
        }
        return res;
    }

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

    private void skipUnprintable() {
        while (hasNextByte() && !isPrintableChar(buffer[ptr]))
            ptr++;
    }

    public boolean hasNext() {
        skipUnprintable();
        return hasNextByte();
    }

    public int nextInt() {
        return Integer.parseInt(next());
    }

    public double nextDouble() {
        return Double.parseDouble(next());
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

    public long min(long... v) {
        long min = Long.MAX_VALUE;
        for (long i : v) min = Math.min(min, i);
        return min;
    }

    public long max(long... v) {
        long max = Long.MIN_VALUE;
        for (long i : v) max = Math.max(max, i);
        return max;
    }
}
