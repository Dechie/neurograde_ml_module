import java.util.*;

public class Main{

	public static void main(String[] args) {
		Scanner scanf =new Scanner (System.in);
		
		int n=scanf.nextInt() , i,S=100000;
		
		for (i=0;i<n;i++)
		{S*=1.05; 
		if(S%1000!=0) S=S+1000-S%1000; }
		System.out.println(S);

	}
}