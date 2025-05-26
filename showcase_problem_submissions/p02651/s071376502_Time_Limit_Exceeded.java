import java.util.Scanner;

public class Main {

  public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    int t = Integer.parseInt(sc.nextLine());
    for (int i = 0; i < t; i++) {
      int n = Integer.parseInt(sc.nextLine());
      long[] a = new long[n];
      String[] attr_a = sc.nextLine().split(" ");
      for(int j = 0; j < n; j++){
        a[j] = Long.parseLong(attr_a[j]);
      }
      String s = sc.nextLine();
      long result = minmax(0, s, a, 0);
      System.out.println(result);
    }
  }
  static long minmax(long cur, String s, long[] arr, int depth){
    if(s.length() <= depth){
      return cur;
    }
    if(s.charAt(depth) == '0'){
      if(cur != 0){
        if(minmax(cur ^ arr[depth], s, arr, depth + 1) == 0){
          return 0;
        } else {
          return minmax(cur , s, arr, depth + 1);
        }
      }
    } else {
      if(cur == 0){
        return minmax(cur ^ arr[depth], s, arr, depth + 1);
      }
      long ret = minmax(cur ^ arr[depth], s, arr, depth + 1);
      if( ret != 0){
        return ret;
      } else {
        return minmax(cur , s, arr, depth + 1);
      }
    }
    return 0;
  }
}