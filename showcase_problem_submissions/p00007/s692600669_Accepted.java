import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;


public class Main {

	public static void main(String[] args) throws IOException {
		try (BufferedReader br = new BufferedReader(new InputStreamReader(
				System.in))) {

			new Main().solution(br);
		}
	}

	private void solution(BufferedReader br) throws IOException {
		int base = 100000;
		int ratio = 5;
		int term = Integer.parseInt(br.readLine());

		int ans = calc(base, ratio, term);
		System.out.println(ans);
	}

	private int calc(int base, int ratio, int term) {
		int result = base;
		int interest;
		int fraction;

		for (int i = 1; i <= term; i++) {
			interest = result / 100 * 5;// ??????
			fraction = interest % 1000;// ?????°
			if (fraction != 0) {
				interest = interest - fraction + 1000;// ??????????????????
			}
			// if(fraction > 500){
			// interest = interest - fraction + 1000;
			// }else{
			// interest = interest - fraction;
			// }
			result += interest;
		}
		return result;
	}
}