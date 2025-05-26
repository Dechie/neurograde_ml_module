import java.util.*;

public class Main {
  public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    int n = sc.nextInt();
    List<List<Integer>> arr = new ArrayList<>();
    for(int i = 0; i < n; i++) {
      arr.add(new ArrayList<>());
      arr.get(i).add(0);
      arr.get(i).add(sc.nextInt());
    }
    int q = sc.nextInt();
    for(int i = 0; i < q; i++) {
      int m = sc.nextInt();
      boolean canMake = false;
      for(int j = 0; j < (int) Math.pow(2 , n); j++) {
        int sum = 0;
        String[] binary = Integer.toBinaryString(j).split("");
        for(int k = 0; k < binary.length; k++) {
          int e = Integer.parseInt(binary[binary.length - 1 - k]);
          sum += arr.get(n - 1 - k).get(e);
        }
        if(sum == m) {
          System.out.println("yes");
          canMake = true;
          break;
        }
      }
      if(!canMake) {
        System.out.println("no");
      }
    }
  }
}

