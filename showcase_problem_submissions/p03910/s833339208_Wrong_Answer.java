import java.util.*;

public class Main {

    public static void main(String args[]) {

        // 入力
        Scanner sc = new Scanner(System.in);
        int n = Integer.parseInt(sc.next());
        sc.close();

        // 主処理
        Deque<String> dq = new ArrayDeque<>();
        while (2 < n) {
            int half = n / 2 + 1;
            n -= half;
            dq.addFirst(Integer.toString(half));
        }
        if (n == 2) {
            dq.addFirst("2");
        } else {
            dq.addFirst("1");
        }

        String result = String.join("\r\n", dq);

        // 出力
        System.out.println(result);
    }
}
