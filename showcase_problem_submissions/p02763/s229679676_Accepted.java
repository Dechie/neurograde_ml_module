import java.util.*;
public class Main {
    public static int max(int a, int b){return (a > b ? a : b);}
    public static int min(int a, int b){return (a < b ? a : b);}
    public static int gcd(int a, int b){return (b > 0 ? gcd(b, a % b) : a);}
    public static int lcm(int a, int b){return a / gcd(a, b) * b;}
    public static void main(String[] args) throws Exception {
        Scanner sc = new Scanner(System.in);
        
        ArrayList<TreeSet<Integer>>S = new ArrayList<TreeSet<Integer>>();
        for(int i = 0; i < 26; i++)S.add(new TreeSet<Integer>());
        for(int i = 0; i < 26; i++)S.get(i).add(505050);
        int N = sc.nextInt();
        String s = sc.next();
        for(int i = 0; i < N; i++){
            int idx = i + 1;
            char c = s.charAt(i);
            int ci = c - 97;
            // System.out.println(idx + " : " + c + " : " + ci);
            S.get(ci).add(idx);
        }
        // System.out.println("");
        int Q = sc.nextInt();
        for(int i = 0; i < Q; i++){
            int q = sc.nextInt();
            if(q == 1){
                int idx = sc.nextInt();
                char c = sc.next().charAt(0);
                int ci = c - 97;
                for(int j = 0; j < 26; j++)S.get(j).remove(idx);
                S.get(ci).add(idx);
                // System.out.println(idx + " : " + c + " : " + ci);
            }
            else{
                int l = sc.nextInt();
                int r = sc.nextInt();
                // System.out.println(l + " - " + r);
                int ans = 0;
                for(int j = 0; j < 26; j++){
                    if(S.get(j).ceiling(l) <= r)ans++;
                }
                System.out.println(ans);
            }
        }
    }
}
