
import java.util.Arrays;
//template code, C343, 2019
import java.util.LinkedList;
import java.util.Scanner;

class Graph2 {
	private int totalVertex;
	private LinkedList<LinkedList<Integer>> adjList;
	public Graph2() { totalVertex = 0; }
	public void loadAdjList() {
		Scanner in = new Scanner(System.in);
		totalVertex = in.nextInt();
		adjList = new LinkedList<LinkedList<Integer>>();
		for(int i = 0; i < totalVertex; i ++) {
			LinkedList<Integer> tmp = new LinkedList<Integer>();
			int idx1 = in.nextInt() - 1;
			int degree = in.nextInt();
			//System.out.println("mark idx1 = " + idx1 + " degree = " + degree);
			for(int j = 0; j < degree; j ++) {
				int idx2 = in.nextInt() - 1;
				tmp.add(idx2);
			}	
			adjList.add(tmp);
		}
		in.close();
	}

	
	static int time = 1;
	
	public void dfs(int idx, int[] discover, int[] finish, int[] visited) {
		
		
		discover[idx] = time;
		time++;
		visited[idx] = 1;
		
		LinkedList<Integer> list = adjList.get(idx);
		
		for (int i = 0; i < list.size(); i++) {
			int next = list.get(i);
			
			if (visited[next] == 0) {
				dfs(next, discover, finish, visited);
			}
			
			
		}
		
		finish[idx] = time;
		time++;
	}
	 
	public int getTotalVertex() {
		return totalVertex;
	}
}

//change class name GraphRepresentation to Main() for submission to AIZU
public class Main {
	public static void main(String argv[]) {
		Graph2 g = new Graph2();
		g.loadAdjList();
		
		int[] discover = new int[g.getTotalVertex()];
		int[] finish = new int[g.getTotalVertex()];
		int[] visited = new int[g.getTotalVertex()];
		
		g.dfs(0, discover, finish, visited);
		
		for (int i = 0; i < g.getTotalVertex(); i++) {
			if (visited[i] == 0) {
				g.dfs(i, discover, finish, visited);
			}
		}
		
		
		for (int i = 0; i < g.getTotalVertex(); i++) {
			System.out.println(i + 1 + " " + discover[i] + " " + finish[i]);
		}
		
		
	}
}

