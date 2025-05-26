
import java.util.*;
public class Main{

	public static void main(String[] args) {
		Scanner  sc=new Scanner(System.in);
	
	          int n=sc.nextInt();
	          int ma=sc.nextInt();
	          int mb=sc.nextInt();
	       int a[]=new int[n];
	       int b[]=new int[n];
	       int c[]=new int[n];
	       for(int i=0;i<n;i++) {
	       	int x=sc.nextInt();
	       	int y=sc.nextInt();
	       	int z=sc.nextInt();
	       	a[i]=x;
	       	b[i]=y;
	       	c[i]=z;
	       }
	       int dp[][][]=new int[n+1][501][501];
	     int max=100000;
	       for(int i=0;i<=n;i++) {
	       	for(int j=0;j<=500;j++) {
	       		for(int k=0;k<=500;k++) {
	       			dp[i][j][k]=max;
	       		}
	       	}
	       }
	       dp[0][0][0]=0;
	       for(int i=1;i<=n;i++) {
	       	for(int j=0;j<=500;j++) {
	       		for(int k=0;k<=500;k++) {
	       			dp[i][j][k]=dp[i-1][j][k];
	       			if(j-a[i-1]>=0 && k-b[i-1]>=0 && dp[i-1][j-a[i-1]][k-b[i-1]]!=max)
	       			dp[i][j][k]=Math.min(	dp[i-1][j-a[i-1]][k-b[i-1]]+c[i-1],dp[i][j][k]);
	       			
	       		}
	       	}
	       }
	       int ans=max;
	      
	       	for(int j=1;j<=500;j++) {
	       		for(int k=1;k<=500;k++) {
	       			if(dp[n][j][k] !=max) {
	       		if((j*mb)==(k*ma)) {
	       			ans=Math.min(ans,dp[n][j][k]);
	       		}
	       		}
	       		}
	       	}
	       if(ans==max) {
	       	System.out.println(-1);
	       }else {
	       	System.out.println(ans);
	       }
	       
	       
	       
	       
	       
	       
	       
	       

		}
	static boolean isValid(int i,int j,int r,int c ) {
			return i>=0&&j>=0&&i<r&&j<c;
	}
	static int[] arr(int n,Scanner sc) {
		int a[]=new int[n];
		for(int i=0;i<n;i++) {
			a[i]=sc.nextInt();
		}
		return a;
}
	}
