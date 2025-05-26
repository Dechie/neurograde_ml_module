import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Main {

	public static void main(String[] args) throws IOException {
		
		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
		
		int n = Integer.parseInt(reader.readLine());
		String[] strArray = reader.readLine().split(" ");
		int[] arr = new int[n];
		for (int i = 0; i < n; i++) {
			arr[i] = Integer.parseInt(strArray[i]);
		}
		
		int q = Integer.parseInt(reader.readLine());
		strArray = reader.readLine().split(" ");
		for (int i = 0; i < q; i++) {
			int target = Integer.parseInt(strArray[i]);
			if (checkIfCanMakeNumber(arr, target)) {
				System.out.println("yes");
			} else {
				System.out.println("no");
			}
		}
	}
	
	private static boolean checkIfCanMakeNumber(int[] arr, int target) {
		return checkIfCanMakeNumber(null, arr, target);
	}

	private static boolean checkIfCanMakeNumber(int[] base, int[] arr, int target) {
		
		boolean ret = false;
		
		if (base != null && base.length > arr.length) {
			return false;
		}
		
		int startPos = 0;
		int baseLen = 0;
		if (base != null) {
			baseLen = base.length;
			startPos = base[baseLen - 1] + 1;
		}
		int[] newBase = new int[baseLen + 1];
		
		for (int i = startPos; i < arr.length; i++) {
			int sum = 0;
			for (int j = 0; j < baseLen; j++) {
				sum += arr[base[j]];
				newBase[j] = base[j];
			}
			sum += arr[i];
			if (sum == target) {
				return true;
			}
			newBase[baseLen] = i;
			ret = checkIfCanMakeNumber(newBase, arr, target);
			if (ret) {
				return ret;
			}
		}
		newBase = null;
		return ret;
	}
}
