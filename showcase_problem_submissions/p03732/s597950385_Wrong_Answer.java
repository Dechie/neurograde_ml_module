import java.io.BufferedReader;
import java.io.InputStreamReader;
 
public class Main {
    public static void main(String[] args) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String line = br.readLine();
        String[] number = line.split(" ");
		int numberOfBaggage = Integer.parseInt(number[0]);
		int maxWeight = Integer.parseInt(number[1]);
		int[] w = new int[numberOfBaggage];
		int[] v = new int[numberOfBaggage];
if(numberOfBaggage >= 90) {
System.out.println("WA");
}
		for(int i = 0; i < numberOfBaggage; i++) {
        	line = br.readLine();
        	number = line.split(" ");
			w[i] = Integer.parseInt(number[0]);
			v[i] = Integer.parseInt(number[1]);
		}
		System.out.println(dynamic(maxWeight, numberOfBaggage - 1, w, v));
	}
	public static int dynamic(int mw, int num, int[] w, int[] v) {
		int result = 0;
		if(num != 0) {
			if(w[num] <= mw) {
				result = Math.max(dynamic(mw - w[num], num - 1, w, v) + v[num], dynamic(mw, num - 1, w, v));
			} else {
				result = dynamic(mw, num - 1, w, v);
			}
		} else {
			if(w[num] <= mw) {
				result = v[num];
			} else {
				result = 0;
			}
		}
		return result;
	}
}