//import java.io.File;
import java.util.Scanner;

class Main{
	public static void main(String[] args) {
try{
		//Scanner scan = new Scanner(new File("in.txt"));
		Scanner scan = new Scanner(System.in);

		int input1 = scan.nextInt();
		int input2 = scan.nextInt();

		boolean[][] map = new boolean[input1][input2];
		for(int i = 0; i < input1; i++) {
			for(int j = 0; j < input2; j++) {
				map[i][j] = scan.nextInt()==0?true:false;
			}
		}

		boolean[][] checkedMap = new boolean[input1][input2];
		int maxSize = 0;
		for(int i = 0; i < input1; i++) {
			for(int j = 0; j < input2; j++) {
//System.out.println("("+i+","+j+")");
				if(!checkedMap[i][j]) {

int dummyMaxSize = 0;
boolean flag = true;
for(int s = 1; flag; s++) {
	if(i+s-1<input1 && j+s-1<input2) {
		if(!map[i+s-1][j+s-1]) {
			for(int _i = i; _i < i+s; _i++) {
				for(int _j = j; _j < j+s; _j++) {
					checkedMap[_i][_j] = true;
				}
			}
			flag = false;
			break;
		}
	}else {
		for(int _i = i; _i < i+s-1; _i++) {
			for(int _j = j; _j < j+s-1; _j++) {
				checkedMap[_i][_j] = true;
			}
		}
		flag = false;
		break;
	}
	for(int w = j; w < j + s; w++) {
		if(w >= input2) break;
		if(!map[i+s-1][w]) {
			for(int _i = i; _i < i+s && _i < input1; _i++) {
				for(int _j = j; _j < w+1; _j++) {
					checkedMap[_i][_j] = true;
				}
			}
			flag = false;
			break;
		}
	}
	for(int h = i; h < i + s; h++) {
		if(h >= input1) break;
		if(!map[h][j+s-1]) {
			for(int _i = i; _i < h+1; _i++) {
				for(int _j = j; _j < j+s && _j < input2; _j++) {
					checkedMap[_i][_j] = true;
				}
			}
			flag = false;
			break;
		}
	}
	if(flag) dummyMaxSize = s;
//System.out.print(flag);
//System.out.println(s);
}
if(maxSize < dummyMaxSize) maxSize = dummyMaxSize;
/*
for(int x = 0; x < input1; x++) {
for(int y = 0; y < input2; y++) {
System.out.print(checkedMap[x][y]?1:0);
}
System.out.println();
}
*/

				}
			}
		}
/*
for(int x = 0; x < input1; x++) {
	for(int y = 0; y < input2; y++) {
		System.out.print(checkedMap[x][y]?1:0);
	}
	System.out.println();
}
*/

		System.out.println(maxSize*maxSize);
		

}catch(Exception e) {
	e.printStackTrace();
}
	}
}