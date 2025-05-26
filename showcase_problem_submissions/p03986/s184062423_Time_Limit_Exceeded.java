import java.io.*;
import java.util.*;
 
public class Main {
    public static void main(String[] args) throws IOException {
		Scanner sc = new Scanner(System.in);
		String a = sc.nextLine();
		a = Main.findSTChars(a);
		System.out.println(a.length());
	}

       static String findSTChars(String str) {
		int len = str.length();
                if (len < 2 || len > 200000)
			return "";
                if (len % 2 != 0)
			return "";
                int S = 0;
		int T = 0;
		for (int i = 0; i < len; i++) {
			if ( str.charAt(i) != 'S' && str.charAt(i) != 'T')
				return "";
			
			if (str.charAt(i) == 'S')
				S++;
			
			if (str.charAt(i) == 'T')
				T++;
		}
		
		if (S != T)
			return "";
		StringBuilder sb = new StringBuilder(str);
		for (int i = 0; i < len - 1; i++) {
			if (sb.charAt(i) == 'S' && sb.charAt(i+1) == 'T') {
				sb.delete(i, i+2);
				len = sb.length();
				i = 0;
			}
		}
		
		for (int i = 0; i < len - 1; i++) {
			if (sb.charAt(i) == 'S' && sb.charAt(i+1) == 'T') {
				sb.delete(i, i+2);
			}
		}
		
		return sb.toString();
	}
}