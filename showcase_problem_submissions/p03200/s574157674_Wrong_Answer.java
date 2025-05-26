import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		String S = sc.nextLine();

		int sum = 0;
		int p = 0;
		for(int i=S.length();i>=1;i--){
			String x = S.substring(i-1, i);
			if(x.equals("W")){
				p+= 1;
			}else{
				sum += p;
			}
		}
		System.out.println(sum);
	}

}

