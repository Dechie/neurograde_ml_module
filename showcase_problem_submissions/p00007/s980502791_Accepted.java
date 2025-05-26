import java.util.*;
import java.lang.*;
import java.math.*;


public class Main {
	
	Scanner sc = new Scanner(System.in);
	
	void run() {
		
		int n = sc.nextInt();
		int b = 100;
		
		for (int i=0;i<n;i++) {
			
			if ( (double)(b*0.05) - (int)(b*0.05) > 0 )
				b = (int)( b * 1.05 ) + 1;
			else 
				b = (int)( b * 1.05 );
			
		}
		
		System.out.println(b*1000);
		
	}
	
	public static void main(String[] args) {
		
		new Main().run();
		
	}
	
}