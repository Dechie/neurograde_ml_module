import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;

/**
 * 0007:Debt Hellの解答
 * 
 * @author MS14A
 * @version 2015/04/15
 */
public class Main {

    /** 初期借金額：10万円 */
    private static final BigDecimal INITIAL_DEBT = new BigDecimal("100000");

    /** 利子：5% */
    private static final BigDecimal INTEREST_RATE = BigDecimal.valueOf(0.05);

    /** \1000未満の繰り上げ計算用 */
    private static final BigDecimal DIVISOR_FOR_ROUND_UP = BigDecimal.valueOf(1000);

    /**
     * n 週間後の借金の残高を出力する。<br>
     * ※n は 100 以下とする。
     * 
     * @param args
     * @throws IOException 
     */
    public static void main(String[] args) throws IOException {
        // 標準入力読み取り
        int weeks = getWeeks();

        // n週間後の借金額計算
        System.out.println(calcDebt(weeks));
    }

    /**
     * 標準入力を読み取り、"n"の値を返す。
     * 
     * @return n：入力値
     * @throws IOException 標準入力読取り失敗時
     */
    private static int getWeeks() throws IOException {
        
        InputStreamReader streamReader = new InputStreamReader(System.in);
        BufferedReader bufferedReader = new BufferedReader(streamReader);
        
        int weeks = Integer.parseInt(bufferedReader.readLine());
        
        bufferedReader.close();
        streamReader.close();
        
        return weeks;
    }

    /**
     * n週間後の借金額を計算する。
     * 
     * @param weeks
     * @return n週間後の借金額
     */
    private static BigDecimal calcDebt(int weeks) {
        BigDecimal finalDebt = INITIAL_DEBT;

        for (int i = 0; i < weeks; i++) {
            // 利子額計算。1000で割った小数部を繰り上げる。
            BigDecimal interestAmount = finalDebt.multiply(INTEREST_RATE)
                    .divide(DIVISOR_FOR_ROUND_UP, 0, BigDecimal.ROUND_UP);

            // 借金額 = 前週の借金額 + 利子額
            finalDebt = finalDebt.add(interestAmount.multiply(DIVISOR_FOR_ROUND_UP));
        }

        return finalDebt;
    }

}