import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class Main {
    Scanner sc = new Scanner(System.in);

    public static final void main(String[] args) {
        new Main().solve();
    }

    void solve() {
        final int h = sc.nextInt();
        final int w = sc.nextInt();
        final int d = sc.nextInt();
        int[][] mat = new int[h][w];
        for (int i = 0; i < h; i++) Arrays.fill(mat[i], -1);
        ArrayList<Integer> xl = new ArrayList<>();
        ArrayList<Integer> yl = new ArrayList<>();
        xl.add(0); yl.add(-d);
        xl.add(-d); yl.add(0);
        for (int i = 1; i <= d; i++) {
            xl.add(i); yl.add(-d+i);
            xl.add(-i); yl.add(-d+i);
        }

        char[] colors = new char[] {'R', 'G'};
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                boolean[] check = new boolean[2];
                for (int k = 0; k < xl.size(); k++) {
                    int x = j + xl.get(k);
                    int y = i + yl.get(k);
                    if (x < 0 || x >= w || y < 0 || y >= h || mat[y][x] == -1) continue;
                    check[mat[y][x]] = true;
                    break;
                }
                for (int k = 0; k < 4; k++) {
                    if (!check[k]) {
                        mat[i][j] = k;
                        sb.append(colors[k]);
                        break;
                    }
                }
            }
            sb.append("\n");
        }
        System.out.println(sb.toString());
    }
}