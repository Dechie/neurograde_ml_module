import java.io.BufferedReader;
import java.io.InputStreamReader;

class Main{
	public static void main(String args[]){
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

		try {
			while(true){
				String strs[] = br.readLine().split(" ");
				int a = Integer.parseInt(strs[0]);
				if(a == 0) return;
				int b = Integer.parseInt(strs[1]);
				int n = Integer.parseInt(strs[2]);
				int count = 0;
				
				int m = a;
				while(true){
					if(check(m)) count++;
					if(count == n){
						System.out.println(m);
						break;
					}
					m += b;
				}
			}
		} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
		}
	}
	
	public static boolean check(int m){
		for(int i = 2; i <= Math.sqrt(m); i++){
			if(m % i == 0) return false;
		}
		return true;
	}
	
	public static void printArray(int array[]){
		for(int i = 0; i < array.length; i++)
			System.out.print(array[i] + "\t");
		System.out.println();
	}

	public static void printArray(boolean array[]){
		for(int i = 0; i < array.length; i++)
			System.out.print(array[i] + "\t");
		System.out.println();
	}
	
	public static void printMatrix(int matrix[][]){
		for(int i = 0; i < matrix.length; i++){
			printArray(matrix[i]);
		}
	}
	
	public static void printMatrix(boolean matrix[][]){
		for(int i = 0; i < matrix.length; i++){
			printArray(matrix[i]);
		}
	}
}

class Coordinates{
	public int value;
	public Coordinates next;
	Coordinates(int input_value){
		value = input_value;
		next = null;
	}
	
	public void add(Coordinates c){
		if(next != null){
			next.add(c);
		}else{
			next = c;
		}
	}
}