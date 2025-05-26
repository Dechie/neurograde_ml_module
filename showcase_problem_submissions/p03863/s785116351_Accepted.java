import java.util.*;

public class Main {
  public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    String s = sc.next();
    int len = s.length();
    int k = 0;
    if(s.charAt(0) == s.charAt(len - 1)) k = 1;
    String ans = "First";
    if(((len - k) % 2) == 0) ans = "Second";
    System.out.println(ans);
  }
}