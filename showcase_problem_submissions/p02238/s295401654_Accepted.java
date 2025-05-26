import java.util.*;

class DFS {
	
	public static int totalVertex;
	public static LinkedList<LinkedList<Integer>> adjList;
	public static int[] d;
	public static int[] f;
	//adjacency list of edges
	public DFS() { 
		totalVertex = 0;
		}
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
		d = new int[totalVertex];
		f = new int[totalVertex];
		for (int i = 0; i < totalVertex; i++) {
			d[i] = -1;
			f[i] = -1;
		}
		in.close();
	}
	
	public static int getDFS(int vertex, int time) {
		LinkedList<Integer> adjacencies = adjList.get(vertex);
	
			d[vertex] = time;
		
			for (int adj : adjacencies) {
				if(d[adj] == -1) {
					time = getDFS(adj, time+1);
				}
			}
		
		
		f[vertex] = time + 1;
		
		return time + 1;
	}
}

public class Main {

	public static void main(String[] args) {
		DFS g = new DFS();
		g.loadAdjList();
		int time = 1;
		for (int i = 0; i < g.totalVertex; i++) {
			if(g.d[i] == -1) {
				time = g.getDFS(i, time) + 1;
			}
		}
		for (int i = 1; i < g.totalVertex + 1; i++) {
			System.out.println(i + " " + g.d[i - 1] + " " + g.f[i - 1]);
		}
	}

}

