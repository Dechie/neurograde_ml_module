import java.util.*;

// CODEFES 2017.qualA-C
// http://code-festival-2017-quala.contest.atcoder.jp/tasks/code_festival_2017_quala_c

public class Main {
	
	static int H;
	static int W;
	static int d;
	
	static boolean solved = false;
	static char[][] map;
	
	public static void main (String[] args) throws InterruptedException {
		Scanner in = new Scanner(System.in);
		
		H = in.nextInt();
		W = in.nextInt();
		d = in.nextInt();
		
		map = new char[H][W];
		
//		d = 2;
//		canPutColor(5, 5, 'c');
		
		solve(0, 0);
		
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				System.out.printf("%c", map[i][j]);
			}
			System.out.println();
		}
	}
	
	public static void solve(int i, int j) {
		if (solved) {
			return;
		}
		if (i == H) {
			solved = true;
			return;
		}
		if (j == W) {
			solve(i + 1, 0);
			return;
		}
//		System.out.printf("i: %d, j: %d\n", i, j);
		char[] color = new char[]{'R', 'Y', 'G', 'B'};
		for (int x = 0; x < color.length; x++) {
			if (canPutColor(i, j, color[x])) {
				map[i][j] = color[x];
				solve(i, j + 1);
			}
		}
	}
	
	// (i, j) is the point where the program wants to place given color
	public static boolean canPutColor(int i, int j, char color) {
		for (int x = i - d; x <= i + d; x++) {
			int up = j - (d - Math.abs(i - x));
			int down = j + (d - Math.abs(i - x));
			
//			System.out.printf("(%d, %d)\n", x, up);
//			if (up != down) {
//				System.out.printf("(%d, %d)\n", x, down);
//			}
			if (0 <= x && x < H && 0 <= up && up < W && map[x][up] == color) {
				return false;
			}
			if (0 <= x && x < H && 0 <= down && down < W && map[x][down] == color) {
				return false;
			}
		}
		return true;
	}
}
