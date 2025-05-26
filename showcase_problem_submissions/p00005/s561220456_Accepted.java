import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		try (Scanner sc = new Scanner(System.in)) {
			while (sc.hasNext()) {
				long x = sc.nextLong();
				long y = sc.nextLong();

				long gcd = gcd(x, y);
				long lcm = (x / gcd) * y;
				System.out.printf("%s %s\n", gcd, lcm);
			}
		}
	}
	public static long gcd(long a, long b) {
		while (a != b) {
			if (a > b) {
				a = a - b;
			} else {
				b = b - a;
			}
		}
		return a;
	}
}