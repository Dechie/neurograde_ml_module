
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Main {
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		String line="";
		while((line = br.readLine())!=null){
			String[] s =line.split(" ");
			long a = Long.parseLong(s[0]);
			long b = Long.parseLong(s[1]);
			if(a>b){
				System.out.println(gcd(b,a)+" "+lcm(b,a));
			}else{
				System.out.println(gcd(a,b)+" "+lcm(a,b));
			}
		}
	}
	public static long gcd(long m, long n){
		if(n == 0)
			return m;
		else
			return gcd(n, m % n);
	}
	public static long lcm(long m, long n){
		return m * n / gcd(m, n);
	}
}