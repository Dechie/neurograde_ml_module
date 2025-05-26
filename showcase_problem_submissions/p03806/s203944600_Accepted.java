import java.util.*;

public class Main {
    static Scanner scanner = new Scanner(System.in);

    public static void main(String[]$) {
        int inf = 1000000;
        int n = scanner.nextInt();
        int ma = scanner.nextInt();
        int mb = scanner.nextInt();

        int[] a = new int[n];
        int[] b = new int[n];
        int[] c = new int[n];

        for (int i = 0; i < n; i++) {
            a[i] = scanner.nextInt();
            b[i] = scanner.nextInt();
            c[i] = scanner.nextInt();
        }

        int[][][] dp = new int[41][401][401];
        Arrays.stream(dp).forEach(a1 -> Arrays.stream(a1).forEach(a2 -> Arrays.fill(a2, inf)));
        dp[0][0][0] = 0;

        for (int i = 0; i < n; i++) {
            for (int ca = 0; ca <= 400; ca++) {
                for (int cb = 0; cb <= 400; cb++) {
                    if (dp[i][ca][cb] != inf) {
                        dp[i + 1][ca][cb] = Math.min(dp[i + 1][ca][cb], dp[i][ca][cb]);
                        dp[i + 1][ca + a[i]][cb + b[i]] = Math.min(dp[i + 1][ca + a[i]][cb + b[i]], dp[i][ca][cb] + c[i]);
                    }
                }
            }
        }

        int ans = inf;
        for (int ca = 1; ca <= 400; ca++)
            for (int cb = 1; cb <= 400; cb++)
                if (ca * mb == cb * ma)
                    ans = Math.min(ans, dp[n][ca][cb]);

        System.out.println(ans == inf ? -1 : ans);
    }
}