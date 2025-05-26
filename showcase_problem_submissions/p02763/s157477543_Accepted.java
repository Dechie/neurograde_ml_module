import java.io.PrintWriter;
import java.util.*;

public class Main {
    static Integer[] seg;
    static int size;
    static void update(int i, int x) {
        i += size;
        seg[i] = x;
        i >>= 1;
        while (i > 0) {
            seg[i] = seg[i * 2] | seg[i * 2 + 1];
            i >>= 1;
        }
    }
    static int query(int l, int r) {
        l += size;
        r += size;
        int s = 0;
        while (l < r) {
            if ((l & 1) == 1) {
                s |= seg[l];
                l++;
            }
            if ((r & 1) == 1) {
                r--;
                s |= seg[r];
            }
            l >>= 1;
            r >>= 1;
        }
        return Integer.bitCount(s);
    }
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        PrintWriter out = new PrintWriter(System.out);
        int n = Integer.parseInt(sc.next());
        size = n;
        seg = new Integer[size * 2];
        Arrays.fill(seg, 0);
        String s = sc.next();
        for (int i = 0; i < n; i++) update(i, 1 << s.charAt(i) - 'a');
        int q = Integer.parseInt(sc.next());
        for (int i = 0; i < q; i++) {
            int p = Integer.parseInt(sc.next());
            if (p == 1) {
                int j = Integer.parseInt(sc.next()) - 1;
                char c = sc.next().charAt(0);
                update(j, 1 << c - 'a');
            }
            else {
                int l = Integer.parseInt(sc.next()) - 1;
                int r = Integer.parseInt(sc.next());
                out.println(query(l, r));
            }
        }
        out.flush();
    }
}