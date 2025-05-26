import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int n = Integer.parseInt(br.readLine());
        String[] a = br.readLine().split(" ");
        int[] A = new int[n];
        for (int i = 0; i< n; i++){
            A[i] = Integer.parseInt(a[i]);
        }
        
        int q = Integer.parseInt(br.readLine());
        String[] b = br.readLine().split(" ");
        int[] Q = new int[q];
        for (int i = 0; i < q; i++){
            Q[i] = Integer.parseInt(b[i]);
        }
        
        long pow = (long) Math.pow(2,n);
        for (int k=0; k<q; k++){
            boolean flag = false;
            for (long i=0; i<pow-1; i++){
                String bits = String.format("%0"+ n + "d", Long.parseLong(Long.toBinaryString(pow-i-1)));
                int tot = 0;
                for (int j=0; j<bits.length(); j++){
                    if(bits.charAt(j)==49){ //"1" == 49
                        tot += A[j];
                    }
                }
                if (tot == Q[k]){
                    System.out.println("yes");
                    flag = true;
                    break;
                }
            }
            if (flag == false){
                System.out.println("no");
            }
        }
    }
}