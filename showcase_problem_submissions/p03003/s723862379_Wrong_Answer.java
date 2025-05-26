
import java.util.Scanner;

public class Main {

	public static void main(String args[])
	{
		Scanner scan=new Scanner(System.in);
		int n=scan.nextInt();
		long k=scan.nextLong();
		int ar[]=new int[n];
		for(int i=0;i<n;i++)
		{
			ar[i]=scan.nextInt();
		}
		long ans=0;
		int sum=0;
		int r=0;
		for(int i=0;i<n;i++)
		{
//			int j=i+1;
			while(sum<k) {
				if(r==n)break;
				sum+=ar[r];
				r++;
			}
			if(sum<k)break;
			ans=(ans+(n-r+1))%1000000007;
			sum-=(ar[i]);
			
			
		}
		System.out.println(ans);
	}
}
