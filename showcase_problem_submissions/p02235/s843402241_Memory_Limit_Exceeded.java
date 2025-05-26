import java.util.Scanner;
class Main{
    final static int  MAX_N = 10000;
    static int[][] memorry = new int[MAX_N][MAX_N];

    public static void main(String[] args){
	Scanner scan = new Scanner(System.in);
	int n;

	n = scan.nextInt();
	for(int i=0; i<n; i++){
	    char[] str1 = scan.next().toCharArray();
	    char[] str2 = scan.next().toCharArray();
	    solve(str1, str2);
	}
    }

    static void solve(char[] str1, char[] str2){
	for(int i=0; i<str1.length; i++){
	    for(int j=0; j<str2.length; j++){
		if(str1[i]==str2[j])memorry[i+1][j+1]=memorry[i][j]+1;
		else memorry[i+1][j+1]=Math.max(memorry[i][j+1], memorry[i+1][j]);
	    }
	}
	System.out.println(memorry[str1.length][str2.length]);
    }
}