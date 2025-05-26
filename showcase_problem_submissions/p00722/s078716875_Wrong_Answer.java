import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

public class Main{
	public static void main(String[] args)throws IOException{
		InputStreamReader isr = new InputStreamReader(System.in);
		BufferedReader reader = new BufferedReader(isr);
		String string;
		int a, d, n, res;
		
		boolean primes[] = new boolean[1000000];
		primes[0] = primes[1] = false;
		Arrays.fill(primes, true);
		for(int i = 2; i * i < 1000000; i++){
			for(int j = 2; j * i < 1000000; j++){
				if(!primes[i * j]) continue;
				primes[i * j] = false;
			}
		}
		
		while(!(string = reader.readLine()).equals("0 0 0")){
			a = Integer.valueOf(string.split(" ")[0]);
			d = Integer.valueOf(string.split(" ")[1]);
			n = Integer.valueOf(string.split(" ")[2]);
			
			res = a;
			for(int i = 0; i < n;){
				if(primes[res += d]) i++;
			}
			System.out.println(res);
		}
		reader.close();
	}
}