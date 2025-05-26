import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Main {
	public static void main(String[] args) throws NumberFormatException, IOException {
		int period = Integer.parseInt(new BufferedReader(new InputStreamReader(System.in), 32).readLine());
		int val = 100;//壱百千円
		for (; 0 < period; period--) {
			val = val * 105 % 100 == 0 ? (int) (val * 1.05) : (int) (val * 1.05) + 1;
		}
		System.out.println(val * 1000);
	}
}