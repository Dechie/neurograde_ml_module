import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String str = sc.next();

        String tmpString = str;
        boolean flag = true;

        if(calMoveable(str)==0){
            flag=true;
        }
        else {
            while (!tmpString.equals("")) {
                tmpString = calNextString(tmpString, flag);
                flag = !flag;
            }
        }

//        int moves = calMoveable(str);
        if(flag){
            System.out.println("Second");
        }
        else{
            System.out.println("First");
        }

    }

    private static String calNextString(String inputStr, boolean isFrist ){
        String tmp = "";
        int moves=0;
        for(int i=1;i<inputStr.length()-1;i++){
            if(inputStr.charAt(i-1)!=inputStr.charAt(i+1)) {
                tmp = inputStr.substring(0, i) + inputStr.substring(i + 1);
                moves = calMoveable(tmp);
                if (isFrist) {
                    if (calMoveable(tmp) % 2 == 1) {
                        if (moves == 0) return "";
                        else return tmp;
                    }
                } else {
                    if (calMoveable(tmp) % 2 == 0) {
                        if (moves == 0) return "";
                        else return tmp;
                    }
                }
            }
        }
        if (moves==0) return "";
        return tmp;
    }

    private static int calMoveable(String input){
        int cnt=0;
        for(int i=1;i<input.length()-1;i++){
            if(input.charAt(i-1)!=input.charAt(i+1)){
                cnt++;
            }
        }
       // System.out.println("move;" + cnt);
        return cnt;
    }

}
