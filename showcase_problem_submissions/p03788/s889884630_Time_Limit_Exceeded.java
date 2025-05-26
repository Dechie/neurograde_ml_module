import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Main {
    public static final String LEFT = "LEFT";
    public static final String RIGHT = "RIGHT";
    public static void main(String[] args) {
        try {
            // 標準入力
            InputStreamReader isr = new InputStreamReader(System.in);
            BufferedReader br = new BufferedReader(isr);
            String nkStr[] = br.readLine().split(" ");
            //String s = br.readLine();
            int n = Integer.parseInt(nkStr[0]);
            int k = Integer.parseInt(nkStr[1]);
            String s = br.readLine();
            String strArr[] = s.split("");
            for (int i = 0; i < k; i++) {
                int now = 0;
                String from = LEFT;
                while (true) {
                    String tmp = strArr[now];
                    if ("A".equals(tmp)) {
                        strArr[now] = "B";
                        if (LEFT.equals(from)) {
                            from = RIGHT;
                            now--;
                        } else {
                            from = LEFT;
                            now++;
                        }
                    } else {
                        strArr[now] = "A";
                        if (LEFT.equals(from)) {
                            from = LEFT;
                            now++;
                        } else {
                            from = RIGHT;
                            now--;
                        }
                    }
                    if (now < 0 || now >= n) {
                        break;
                    }
                }
            }
            for (String str : strArr) {
                System.out.print(str);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}