import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;
import java.util.Arrays;

import static java.lang.Integer.parseInt;

/**
 * Problem A: Dirichlet's Theorem on Arithmetic Progressions
 */
public class Main {
	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		String line;

		boolean[] ps = new boolean[1000000 + 1];
		Arrays.fill(ps, true);

		for (int i = 2; i < ps.length; i++) {
			if (ps[i]) {
				for (int j = i * 2; j < ps.length; j += i) {
					ps[j] = false;
				}
			}
		}

		while ((line = br.readLine()) != null && !line.isEmpty()) {

			StringTokenizer st = new StringTokenizer(line);

			int a, d, n;
			a = parseInt(st.nextToken());
			d = parseInt(st.nextToken());
			n = parseInt(st.nextToken());
			if ((a | d | n) == 0) break;

			int i = a, c = 0;
			while (true) {
				if (ps[i]) {
					c++;
					if (c == n) break;
				}
				i += d;
			}
			System.out.println(i);

		} // end while
	} // end main
}