import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.StringTokenizer;

public class Main {
    BufferedReader br;
    PrintWriter out;
    StringTokenizer stok;

    String nextToken() throws IOException {
        while (stok == null || !stok.hasMoreTokens()) {
            String s = br.readLine();
            if (s == null) {
                return null;
            }
            stok = new StringTokenizer(s);
        }
        return stok.nextToken();
    }

    int nextInt() throws IOException {
        return Integer.parseInt(nextToken());
    }

    long nextLong() throws IOException {
        return Long.parseLong(nextToken());
    }

    double nextDouble() throws IOException {
        return Double.parseDouble(nextToken());
    }

    char nextChar() throws IOException {
        return (char) (br.read());
    }

    String nextLine() throws IOException {
        return br.readLine();
    }

    boolean inField(int x, int y, int n, int m) {
        return (x >= 0 && y >= 0 && x < n && y < m);
    }

    class Pair {
        int x, y, tmp;
        boolean c;

        public Pair(int x, int y, boolean c) {
            this.x = x;
            this.tmp = -1;
            this.y = y;
            this.c = c;
        }
    }

    int[] dx = {1, -1, 1, -1};
    int[] dy = {1, 1, -1, -1};

    boolean nei(Pair a, Pair b, int d) {
        return (Math.abs(a.x - b.x) + Math.abs(a.y - b.y) == d);
    }

    Integer[] mrandom(int x) {
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = 1; i < 5; i++) {
            if (i != x) list.add(i);
        }
        Integer[] xx = new Integer[0];
        return list.toArray(xx);
    }

    void solve() throws IOException {
        int n = nextInt();
        int m = nextInt();
        int d = nextInt();
        int[][] a = new int[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (a[i][j] == 0) a[i][j] = 1;;

                ArrayList<Pair> list = new ArrayList<>();
                for1:
                for (int di = 0; di <= d; di++) {
                    int dj = d - di;
                    for (int v = 0; v < 4; v++) {
                        int ni = i + dx[v] * di;
                        int nj = j + dy[v] * dj;
                        if (inField(ni, nj, n, m)) {
                            list.add(new Pair(ni, nj, a[ni][nj] != 0));
                        }
                    }
                }
                Collections.shuffle(list);
                for (int k = 0; k < list.size() - 1; k++) {
                    for (int l = k + 1; l < list.size(); l++) {
                        if (nei(list.get(k), list.get(l), d)) {
                            if (list.get(k).tmp != -1) {
                                a[list.get(l).x][list.get(l).y] = 10 - a[i][j] - list.get(k).tmp - a[list.get(k).x][list.get(k).y];
                            } else {
                                Integer[] pp = mrandom(a[i][j]);
                                a[list.get(l).x][list.get(l).y] = (int) pp[0];
                                a[list.get(k).x][list.get(k).y] = (int) pp[1];
                                list.get(k).tmp = pp[2];
                                list.get(l).tmp = pp[2];
                            }
                        }
                    }
                }
                for (int k = 0; k < list.size(); k++) {
                    if (a[list.get(k).x][list.get(k).y] == 0) {
                        Integer[] pp = mrandom(a[i][j]);
                        a[list.get(k).x][list.get(k).y] = (int) pp[0];
                    }
                }

            }
        }
        char[] c = "RYGB".toCharArray();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                out.print(c[a[i][j] - 1]);
            }
            out.println();
        }
    }

    void run() throws IOException {
        // br = new BufferedReader(new FileReader("dragons.in"));
        //out = new PrintWriter("dragons.out");
        // br = new BufferedReader(new FileReader("input.txt"));
        // out = new PrintWriter("output.txt");
        br = new BufferedReader(new InputStreamReader(System.in));
        out = new PrintWriter(System.out);
        solve();
        br.close();
        out.close();
    }

    public static void main(String[] args) throws IOException {
        // Locale.setDefault(Locale.US);
        new Main().run();
    }
}