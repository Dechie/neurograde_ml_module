import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner in = new Scanner(new BufferedReader(new InputStreamReader(System.in)));
		String s = in.next();
		System.out.println(solve(s));
		in.close();
	}

	private static int solve(String s) {
		int ra = 1;
		int re = 1;
		for (int i = 0; i < s.length(); i++) {
			ra *= 3;
			if (s.charAt(i) == '1')
				re *= 2;
			else
				ra -= re * 2;
			ra %= 1000000007;
			re %= 1000000007;
		}

		return ra;
	}
}