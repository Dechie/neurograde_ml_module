import java.util.*;
class Main{
    public static void main(String[] $){
        Scanner s=new Scanner(System.in);
        String c=s.next();
        int a=0, w=0;
        for(int i=0;i<c.length();i++) if(c.charAt(i)=='W') a+=i-w++;
        System.out.print(a);
    }
}