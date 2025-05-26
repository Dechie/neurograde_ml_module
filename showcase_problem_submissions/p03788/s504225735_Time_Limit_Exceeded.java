import java.util.Scanner;
public class Main {
	public static void main(String[] args){
		Scanner as = new Scanner(System.in);
		int n = as.nextInt();
		int k = as.nextInt();
		String a = as.next();
		int[] b;
		b = new int[n];
		for(int i=0;i<n;i++){
			if(String.valueOf(a.charAt(i)).equals("A")){
				b[i] = 1;
			}else{
				b[i] = -1;
			}
		}
		for(int i=0;i<k;i++){
			if(b[0]==1){
				b[0]*=-1;
			}else{
				for(int s=0;s<n-1;s++){
					b[s]=b[s+1]*-1;
				}b[n-1]=1;
			}
		}
		String l = "";
		for(int i=0;i<n;i++){
			if(b[i]==1){
				l+="A";
			}else{
				l+="B";
			}
		}
		System.out.println(l);
	}
}