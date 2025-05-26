import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main {

	public static void main(String[] args) throws Exception {
		// TODO ?????????????????????????????????????????????
		BufferedReader br =
				new BufferedReader(new InputStreamReader(System.in));

		int n = Integer.parseInt(br.readLine());

		adjmatrix = new int[n+1][n+1];
//		issearched = new boolean[n+1];
		d = new int[n+1];
		f = new int[n+1];

		for (int i=0; i < n; i++) {
			String[] list = br.readLine().split(" ");
			int u = Integer.parseInt(list[0]);
			int k = Integer.parseInt(list[1]);
			adjmatrix[u][0] = u;
			for (int j=0; j < k; j++) {
				int vi = Integer.parseInt(list[j+2]);
				adjmatrix[u][vi] = 1;
			}
		}
//		search(adjmatrix[1]);
//		System.out.println(Arrays.toString(adjmatrix[2]));
		for (int i=1; i < n; i++) {
			search(adjmatrix[i]);
		}
		for (int i=1; i <= n; i++) {
			System.out.println(i + " " + d[i] + " " + f[i]);
		}
	}

	static int time = 0;

	static int[] d;

	static int[] f;

	static int[][] adjmatrix;

//	static boolean[] issearched;

	public static void search(int[] adjlist) {
		int u = adjlist[0];
//		issearched[u] = true;
		if (d[u] != 0) return;

		time++;
		d[u] = time;

		int n = adjlist.length - 1;
		for (int i=1; i <= n; i++) {
			if (adjlist[i] == 1) {
				search(adjmatrix[i]);
			}
		}

		time++;
		f[u] = time;
	}
}