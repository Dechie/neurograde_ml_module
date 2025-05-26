import java.util.*;
public class Main {
	private static int ans[];
	public static void main(String[] args){
    	Scanner sc = new Scanner(System.in);
    	int n = sc.nextInt();
    	String s = sc.next();
    	int q = sc.nextInt();

    	TreeSet[] segs = new TreeSet[26];
    	char[] d = s.toCharArray();
    	for (int i=0;i<26;i++){
    		segs[i]=new TreeSet();
    		segs[i].add(510000);
    	}
    	for (int i=0;i<n;i++){
    		segs[(int)d[i]-'a'].add(i);
    	}
    	for (int i=0;i<q;i++) {
    		int a = sc.nextInt();
    		if (a==1) {
    			int index = sc.nextInt()-1;
    			char oldc = d[index];
    			char newc = sc.next().toCharArray()[0];
    			if (oldc == newc) continue;
    			d[index]=newc;
    			segs[(int)oldc-'a'].remove(index);
    			segs[(int)newc-'a'].add(index);
    			continue;
    		}else{
	            int l = sc.nextInt()-1;
	            int r = sc.nextInt()-1;
	            int ans = 0;
	            for (int j = 0; j < 26; j++) {
	                if ((int)segs[j].ceiling(l) <= r) {
	                    ans += 1;
	                }
	            }
	            System.out.println(ans);
    		}
    	}
    }
} 

