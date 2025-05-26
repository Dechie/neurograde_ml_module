import java.util.Scanner;

public class Main {
	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);
		int N = Integer.parseInt(scan.next());
		long K = Long.parseLong(scan.next());
		byte[] str = scan.next().getBytes();
		for (int i = 0; i < K; i++) {
			if (str[0] == 'A') {
				str[0] = 'B';
				continue;
			}
			for (int j = 0; j < N - 1; j++) {
				str[j] = (byte) ((str[j + 1] == 'A') ? 'B' : 'A');
			}
			str[N - 1] = 'A';
		}
		for (byte temp : str) {
			System.out.print((char) temp);
		}
	}
}
