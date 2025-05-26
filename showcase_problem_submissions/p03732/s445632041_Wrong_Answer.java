import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.Scanner;

/**
 * Built using CHelper plug-in
 * Actual solution is at the top
 */
public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        Scanner in = new Scanner(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        SimpleKnapsack solver = new SimpleKnapsack();
        solver.solve(1, in, out);
        out.close();
    }

    static class SimpleKnapsack {
        public void solve(int testNumber, Scanner in, PrintWriter out) {
            int n = in.nextInt();
            int w = in.nextInt();

            int[] weights = new int[n];
            int[] values = new int[n];

            // i番目 | 選んだ個数 | 残 weight
            long[][][] dp = new long[n + 1][n + 1][3 * n + 1];
            for (int i = 0; i < n; i++) {
                int ww = Integer.parseInt(in.next());
                int vv = Integer.parseInt(in.next());
                weights[i] = ww;
                values[i] = vv;
            }
            int limit = w - (w / weights[0]) * weights[0];

            for (int i = 0; i < n; i++) {
                int diff = weights[i] - weights[0];
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k <= 3 * n; k++) {
                        if (k - diff < 0) {
                            dp[i + 1][j + 1][k] = dp[i][j + 1][k];
                        } else {
                            dp[i + 1][j + 1][k] = Math.max(dp[i][j + 1][k], dp[i][j][k - diff] + values[i]);
                        }
                    }
                }
            }
            long result = 0;
            for (int i = 0; i <= n; i++) {
                long zan = w - i * weights[0];
                if (zan >= 0) {
                    result = Math.max(result, dp[n][i][(int) Math.min(3 * n, zan)]);
                }
            }

            out.println(result);

        }

    }
}

