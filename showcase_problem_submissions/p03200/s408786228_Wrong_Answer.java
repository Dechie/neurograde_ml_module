import java.util.*;
import java.lang.*;


public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String s = sc.next();
        long count = 0;
        long wcount = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == 'W' && i > wcount) {
                count += i - wcount;
                wcount++;
            }
        }
        System.out.println(count);
    }
}