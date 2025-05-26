import java.util.Scanner;

public class Main {

  public static final String FIRST = "First";
  public static final String SECOND = "Second";

  public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    String s = sc.next();
    sc.close();

    String winner = SECOND;
    for (int i = 1; i < s.length() - 1; i++) {
      if (s.charAt(i - 1) != s.charAt(i + 1)) {
        s = s.substring(0, i) + s.substring(i + 1);
        i -= 2;
        if (i < 0) {
          i = 0;
        }
        winner = winner == FIRST ? SECOND : FIRST;
      }
    }
    System.out.println(winner);
  }
}
