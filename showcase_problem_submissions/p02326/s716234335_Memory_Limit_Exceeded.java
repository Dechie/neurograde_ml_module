import java.util.*;

public class Main {/* Largest Square */
	
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int h, w, maxS = 0;
		
		h = sc.nextInt();
		w = sc.nextInt();

		int[][] tiles = new int[h][w], map = new int[h][w];
		for(int i = 0; i < h; i++){
			for(int j = 0; j < w; j++){
				tiles[i][j] = sc.nextInt();/* ???????????? */
				map[i][j] = -1;/* ?????£????????\??????????????? */
			}
		}
		
		maxS = 0;
		for(int i = 0; i < h; i++){
			for(int j = 0; j < w;  j++){
				int ret = search(tiles, i, j, map);
				ret *= ret;
				if(ret > maxS){
					maxS = ret;
				}
			}
		}
		sc.nextLine();
		System.out.println(maxS);		
		}
	
	public static int search(int[][] tiles, int i, int j, int[][] map){
		if(tiles[i][j] == 1){
			map[i][j] = 0;
			return 0;
		}else if(map[i][j] != -1){
			return map[i][j];
		}else{
			if(i + 1 < tiles.length && j + 1 < tiles[0].length){
				int min = search(tiles, i+1, j, map);
				int ret = search(tiles, i, j+1, map);
				if(ret < min){
					min = ret;
				}
				ret = search(tiles, i+1, j+1, map);
				if(ret < min){
					min = ret;
				}
				map[i][j] = min + 1;
				return map[i][j];				
			}else{
				map[i][j] = 1;
				return 1;
			}
		}
	}
}