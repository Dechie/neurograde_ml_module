import java.util.*;

public class Main {
   public static void main(String[] args) {
       Scanner sc = new Scanner(System.in);
 while(sc.hasNext()){
	 long n1=sc.nextLong();
	 long n2=sc.nextLong();
	 long b=n1*n2;
	  n1= Math.max(n1, n2);
	  n2= Math.min(n1, n2);
	 long a; 
	  for(;;){
	 a=n1%n2;
	 n1=n2;
	 n2=a;
	 if(a==0)break; 
	 }
	 System.out.println(n1+" "+(b/n1));
	}
 }
   }