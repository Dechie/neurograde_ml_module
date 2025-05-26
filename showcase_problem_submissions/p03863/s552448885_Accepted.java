import java.util.Scanner;

public class Main {

  public static final String FIRST = "First";
  public static final String SECOND = "Second";

  public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    String s = sc.next();
    sc.close();

    int count = 0;
    if (s.charAt(0) == s.charAt(s.length() - 1)) {
      count = s.length() - 3;
    } else {
      count = s.length() - 2;
    }
    String result = count % 2 == 1 ? FIRST : SECOND;
    System.out.println(result);
  }
}
