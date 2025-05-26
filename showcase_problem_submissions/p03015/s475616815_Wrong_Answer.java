import java.util.Scanner;
public class Main{
	public static void main(String[] args){
		Scanner sc = new Scanner(System.in);
		String l=sc.next();
		long mod=1000000007;
		// dp[][1]の時小さいことが確定
		long[][] dp=new long[l.length()+1][2];
		dp[0][0]=1;
		for(int i=1; i<=l.length(); i++){
			if(l.charAt(i-1)=='1'){
				dp[i][0]=(dp[i-1][0]*2)%mod;
				dp[i][1]=dp[i-1][0];
			}else{
				dp[i][0]=dp[i-1][0];
			}
			dp[i][1]+=(dp[i-1][1]*3)%mod;
			dp[i][1]%=mod;
		}
		System.out.println(dp[l.length()][1]+dp[l.length()][0]);
	}
}


