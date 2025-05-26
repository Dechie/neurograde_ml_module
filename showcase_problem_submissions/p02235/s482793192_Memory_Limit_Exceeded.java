import java.util.Scanner;
import java.util.Arrays;
public class Main{
    static int dp[][]=new int[10000][10000];
	public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int num = sc.nextInt();
        int ans[]=new int[num];
        for(int i=0;i<num;i++){
            String str1=sc.next();
            String str2=sc.next();
            for(int x=1;x<=str2.length();x++){
                for(int y=1;y<=str1.length();y++){
                    String temp1=String.valueOf(str1.charAt(y-1));
                    String temp2=String.valueOf(str2.charAt(x-1));
                    if(temp1.equals(temp2)){
                        dp[x][y]=dp[x-1][y-1]+1;
                    }
                    else{
                        if(dp[x-1][y]>dp[x][y-1]){
                            dp[x][y]=dp[x-1][y];
                        }
                        else{
                            dp[x][y]=dp[x][y-1];
                        }
                    }
                }
            }
            ans[i]=dp[str2.length()][str1.length()];
        }
        for(int i=0;i<num;i++){
            System.out.println(ans[i]);
        }
    }
}