import java.util.*;

/**
 *
 */
public class Main {
    public static final int MOD = 1_000_000_007;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        String L = sc.next();

        System.out.println(solveNonDP(L));
    }

    /**
     * DPではない解法
     * yana87cp さんの https://atcoder.jp/contests/abc129/submissions/5857293
     * tempura_cpp さんの https://atcoder.jp/contests/abc129/submissions/5857001
     * catupper さんの https://www.youtube.com/watch?v=FI_Kl9V8JaU
     * @param L
     * @return
     */
    private static int solveNonDP(String L) {
        long ans = 0;
        int oneCount = 0;
        int N = L.length();

        // A+B < Lのケース
        for (int i=0; i<N; i++) {
            // Lの上からi+1桁目でA+Bが下回ることが決定するとき
            // i+1桁目は1であることが前提 (0だとA+B=Lなので下回れない)
            if (L.charAt(i) == '0') continue;

            // (i桁目まではLと等しい組合せ数) * ( i+1桁は0の1通り) * (i+2桁以降は任意)
            // = 2^oneCount (詳細はA+B==Lのケース参照) * 1 * (i+2桁以降のN-i-1桁について，A=B=0, A=1/B=0, A=0,B=1の3パターン)
            // = 2^oneCount * 3^(N-i-1)
            ans += modPow(2, oneCount, MOD) * modPow(3, N-i-1, MOD) % MOD;
            ans %= MOD;

            oneCount++;
        }

        // A+B == Lのケース
        // Lの各ビットに対して，1ビットがAかBのどちらか(もう一方は0)の2パターンなので
        // 2^oneCount
        ans += modPow(2, oneCount, MOD);
        ans %= MOD;

        return (int)ans;
    }

    private static long modPow(int a, int b, int m) {
        long value = 1;

        while (b > 0) {
            if (b%2 != 0) {
                value *= a;
                value %= m;
                b -= 1;
            } else {
                b /= 2;
                a *= a;
                a %= m;
            }
        }

        return value;
    }
}