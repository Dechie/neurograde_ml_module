import java.util.*;

class Main{
	public static void main(String[] $){
		Scanner s=new Scanner(System.in);
		int n=s.nextInt(),l=(int)(Math.sqrt(n*4)),a=l*(l+1)/2;
		BitSet b=new BitSet(n);
		b.set(1,l);
		for(int i=l;i>0;--i){
			if(a-i>=n){
				a-=i;
				b.clear(i);
			}
		}
		b.stream().forEach(System.out::println);
	}
}