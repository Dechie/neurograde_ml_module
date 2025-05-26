import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
class Main{
    List<Integer> A = new ArrayList<Integer>();
    public void run(){
        try(BufferedReader br = new BufferedReader(new InputStreamReader(System.in))){
            int digit = Integer.parseInt(br.readLine());
            List<String> list = createByte(digit);
            Arrays.stream(br.readLine().split(" "))
                .forEach(n -> A.add(Integer.parseInt(n)));
            String strArray = br.readLine();
            Arrays.stream(br.readLine().split(" "))
                .forEach(n -> System.out.println(checkContain(A, list, Integer.parseInt(n), digit)));
        }catch(Exception e){
        }
    }

    public List<String> createByte(int digit){
        List<String> list = new ArrayList<>();
        for(int i = 1; i < (int)(Math.pow(2.0, (double)digit)); i++){
            list.add(Integer.toBinaryString(i));
        }
        return list;
    }

    public String checkContain(List<Integer> A, List<String> list, int target, int digit){
        int sum = 0;
        for(String a : list){
            sum = 0;
            for(int j = a.length() ; j < digit;j++)
                a = "0"+ a;
            String[] tmp = a.split("");
            for(int i = 0;i < digit; i++){
                if(Integer.parseInt(tmp[i]) == 1)
                    sum += A.get(i);
                if(sum > target){
                    sum = 0;
                    break;
                }
                else if(sum == target){
                    return "yes";
                }
            }
        }
        return "no";
    }

    public static void main(String[] args){
        new Main().run();
    }
}