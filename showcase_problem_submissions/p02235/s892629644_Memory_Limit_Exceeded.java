import java.io.*;

public class Main {
	public static void main(String[] args) throws IOException{
		final BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
		int n=Integer.parseInt(br.readLine());
		for(int i=0;i<n;i++){
			int line = 0;
			String str1=br.readLine();
			String str2=br.readLine();
			int renge1=str1.length();
			int renge2=str2.length();
			int[][] x=new int[renge1+1][renge2+1];
			str1=" "+str1;
			str2=" "+str2;	
			for(int j=1;j<=renge1;j++){
				for(int k=1;k<=renge2;k++){
					String a=str1.substring(j, j+1);
					String b=str2.substring(k,k+1);
					if(a.equals(b)) x[j][k]=x[j-1][k-1]+1;
					else x[j][k]=Math.max(x[j-1][k],x[j][k-1] );
					line=Math.max(line, x[j][k]);
				}
				
			}
			System.out.println(line);
		}				
	}
}