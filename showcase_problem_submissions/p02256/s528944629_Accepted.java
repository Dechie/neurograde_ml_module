


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

class Main{
  public static void main(String args[]) throws IOException{
   BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

   String[] string =  br.readLine().split(" ");

   int x = Integer.parseInt( string[0] );
   int y = Integer.parseInt( string[1] );

   int tmp;

   if(x < y){

	   tmp = x;
	   x = y;
	   y = tmp;

   }

   int r = x % y;
   while(r!=0){
     x = y;
     y = r;
	 r = x % y;
   }

   System.out.println(y);

 }
}