import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Scanner;

class Main {

	static long changecount = 0;
   public static void main(String args[]) throws IOException{

	   Scanner sc = new Scanner(System.in);
	   BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
	   int n = Integer.parseInt( br.readLine()  );


	   String[] A = new String[n];
	   String[] origin = new String[n];
	   for(int i = 0;i < n; i++){
          A[i] =  br.readLine();
          origin[i] = A[i];
	   }



	   //int q = partition(A , 0 , n-1);
	   quicksort(A , 0 , n-1);


	   int Choicecheck = 1;
	   for(int i =0; i < n; i++){
		   for(int j = i+1; j < n; j++){
			   for(int a = 0; a < n; a++){
				   for(int b =a+1; b<n;b++){
					  // String[] card =  origin[i].split(" ");
					   //String[] card2=  origin[j].split(" ");


					   if(  origin[i].split(" ")[1].equals(origin[j].split(" ")[1]) && origin[i].equals(A[b]) && origin[j].equals(A[a])){
						   Choicecheck = 0;


					   }

				   }
			   }
		   }
	   }

	   if(Choicecheck == 1){
		   System.out.println("Stable");
	   }else{
		   System.out.println("Not stable");
	   }




	   for(int i = 0;i < n; i++){

		   System.out.println(A[i]);

	   }
	   //System.out.println("");
	   sc.close();
   }

   public static int partition(String[] A, int p, int r){

	  int i = p-1;

       int x = Integer.parseInt( A[r].split(" ")[1] )  ;


      for(int j = p; j < r; j++){

    	  int number = Integer.parseInt( A[j].split(" ")[1] )  ;
    	  if(number <= x){
    		  i++;
    		  String tmp = A[i];
    		  A[i] = A[j];
    		  A[j] = tmp;
    	  }

      }


      String tmp = A[i+1];
	  A[i+1] = A[r];
	  A[r] = tmp;

	  return i+1;
   }


   public static void quicksort(String[] A, int p, int r){

    if(p < r){
    	//System.out.println( "quicksort" );
        int q = partition(A, p, r);
        quicksort(A, p, q-1);
        quicksort(A, q+1, r);
    }


  }

}