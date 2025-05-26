import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.StringTokenizer;
//https://stackoverflow.com/questions/31065929/number-of-common-sub-sequences-of-two-strings
//https://www.geeksforgeeks.org/count-common-subsequence-in-two-strings/

class Reader {
    static BufferedReader reader;
    static StringTokenizer tokenizer;

    /** call this method to initialize reader for InputStream */
    static void init(InputStream input) {
        reader = new BufferedReader(new InputStreamReader(input) );
        tokenizer = new StringTokenizer("");
    }

    /** get next word */
    static String next(
    		) throws IOException {
        while ( ! tokenizer.hasMoreTokens() ) {
            //TODO add check for eof if necessary
            tokenizer = new StringTokenizer(reader.readLine() );
        }
        return tokenizer.nextToken();
    }

    static int nextInt() throws IOException {
        return Integer.parseInt( next() );
    }
	
    static double nextDouble() throws IOException {
        return Double.parseDouble( next() );
    }
    static long nextLong() throws IOException {
        return Long.parseLong( next() );
    }
}

public class Main {
	
	static int mod=1000000007;
	
	public static void main(String[] args) throws IOException{
		Reader.init(System.in);
		int N=Reader.nextInt();
		int M=Reader.nextInt();
		int S[]=new int[N];
		int T[]=new int[M];
		for(int i=0;i<N;i++) {
			S[i]=Reader.nextInt();
		}
		for(int i=0;i<M;i++) {
			T[i]=Reader.nextInt();
		}
		int result=commonSubsequence(S,T, N, M);
		System.out.println(result);
	}

	private static int commonSubsequence(int[] s, int[] t, int n, int m) {
		int dp[][]=new int[n+1][m+1];
		
		for(int i=0;i<=n;i++) {
			dp[i][0]=1;
		}
		for(int j=0;j<=m;j++) {
			dp[0][j]=1;
		}
		
		for(int i=1;i<=n;i++) {
			for(int j=1;j<=m;j++) {
				if(s[i-1]==t[j-1]) {
					dp[i][j]=dp[i][j-1]+dp[i-1][j]+mod;
				}
	            else {
	            	dp[i][j]=dp[i][j-1]+dp[i-1][j]-dp[i-1][j-1]+mod;
	            }
				dp[i][j]%=mod;
			}
		}
		return dp[n][m];
	}
	
}