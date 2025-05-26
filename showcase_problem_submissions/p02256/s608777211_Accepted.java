import java.io.*;

public class Main {
    public static void main(String[] args) {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        try {
            String[] nums = br.readLine().split(" ");
            int x = Integer.parseInt(nums[0]);
            int y = Integer.parseInt(nums[1]);

            while(y != 0) {
                int tmp = y;
                y = x % y;
                x =tmp;
            }
            System.out.println(x);
        } catch (IOException e) {
            System.out.println(e);
        }
    }
}