import java.util.Scanner;


public class Main{

	public static void main(String args[]){

		long MOD=1000000007L;

		Scanner sc = new Scanner(System.in);

		String s=sc.next();
		char c[]=s.toCharArray();
		int k=c.length;
		int count=0;
		long p3[]=new long[k+1];
		p3[0]=1;
		for(int i=1;i<=k;++i){
			p3[i]=p3[i-1]*3%MOD;
		}

		long ans=0L;

		for(int i=0;i<k;++i){
			if(c[i]=='1'){
				ans = (ans + p3[k-i-1]*(long)Math.pow(2, count))%MOD;
				++count;
			}
		}
		ans+=(long)Math.pow(2, count);
		System.out.println(ans);
	}
}