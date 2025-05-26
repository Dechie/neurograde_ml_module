import java.util.*;

public class Main {
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		String s = sc.next();
		int counter = 0;
		boolean next = true;
		while(next) {
			if(s.length() == 2)
				next = false;
			else {
				for(int i = 1; i <= s.length() - 2; i++) {
					if(s.charAt(i - 1) != s.charAt(i + 1)) {
						s = s.substring(0, i) + s.substring(i + 1, s.length());
						break;
					} else if(i == s.length() - 2)
						next = false;
				}
			}
			counter++;
		}
		System.out.println((counter%2 == 0) ? "First" : "Second");
	}
}