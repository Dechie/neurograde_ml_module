import java.util.Scanner;
public class Main {
	public static void main(String[] args) {
		ABC054DSolve solve = new ABC054DSolve();
		solve.main();
	}
}

class ABC054DSolve {
	
	int N;
	int Ma;
	int Mb;
	Drug[] drugs;
	
	int maxA = 0;
	int maxB = 0;
	int[][][] dp;
	
	int INF = Integer.MAX_VALUE / 2;
	
	ABC054DSolve() {
		Scanner cin = new Scanner(System.in);
		this.N = cin.nextInt();
		this.Ma = cin.nextInt();
		this.Mb = cin.nextInt();
		this.drugs = new Drug[N];
		
		for (int i = 0; i < N; i++) {
			int a = cin.nextInt();
			int b = cin.nextInt();
			int cost = cin.nextInt();
			drugs[i] = new Drug(a, b, cost);
			
			maxA += a;
			maxB += b;
		}
	}
	
	void main() {
		
		// initialize dp
		this.dp = new int[N][maxA + 1][maxB + 1];
		for (int k = 0; k < N; k++) {
			for (int i = 0; i <= maxA; i++) {
				for (int j = 0; j <= maxB; j++) {
					dp[k][i][j] = INF;
				}
			}
		}
		dp[0][0][0] = 0;
		dp[0][drugs[0].a][drugs[0].b] = drugs[0].cost;
		
		// solve recurrence relation
		for (int i = 0; i < N; i++) {
			for (int x = 0; x <= maxA; x++) {
				for (int y = 0; y <= maxB; y++) {
					if (i - 1 >= 0) {
						dp[i][x][y] = dp[i - 1][x][y];
						if (x - drugs[i].a >= 0 && y - drugs[i].b >= 0) {
							dp[i][x][y] = Math.min(dp[i][x][y], dp[i - 1][x - drugs[i].a][y - drugs[i].b] + drugs[i].cost);
						}
					}
				}
			}
		}
		
		
		int minCost = Integer.MAX_VALUE / 2;
		for (int i = 1; i <= maxA; i++) {
			for (int j = 1; j <= maxB; j++) {				
				if (i*Mb == j*Ma) {
					minCost = Math.min(minCost, dp[N - 1][i][j]);
				}
			}
		}
		System.out.println(minCost != INF ? minCost : -1);	
	}
	
	class Drug {
		int a;
		int b;
		int cost;
		
		Drug(int a, int b, int cost) {
			this.a = a;
			this.b = b;
			this.cost = cost;
		}
	}
	
}
