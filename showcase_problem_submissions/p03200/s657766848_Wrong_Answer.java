import java.util.*;

class Main {
    public static void main(String[] args) {
        new Solver().run();
    }
}

class Solver {

    public void run() {
        Scanner sc = new Scanner(System.in);
        char[] S = sc.next().toCharArray();

        int cnt = 0;
        for (int j = S.length-1; j >= 0; ) {
            int right = j;
            int left = right;
            boolean found = false;
            // find right 'W'
            for (int i = j; i >= 0; i--) {
                char ch = S[i];
                if (ch == 'W') {
                    right = i;
                    found = true;
                    break;
                }
            }
            if (!found) break;

            // find left 'B'
            found = false;
            for (int i = right - 1; i >= 0; i--) {
                char ch = S[i];
                if (ch == 'B') {
                    left = i;
                    found = true;
                    break;
                }
            }

            if (!found) break;
            cnt += (right - left);
            //System.out.println(String.format("%d %d %d", cnt, left, right));
            // swap
            S[left] = 'W';
            S[right] = 'B';
            j = right - 1;
        }


        System.out.println(cnt);

        return;
    }

}