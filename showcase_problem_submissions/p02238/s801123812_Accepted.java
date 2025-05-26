import java.util.*;

public class Main {

    static class Edge {
        int from, to;

        public Edge(int from, int to) {
            this.from = from;
            this.to = to;
        }
    }

    static void dfs(int at) {
        if (visited[at]) return;
        visited[at] = true;
        d[at] = timeStamp++;

        for (Edge e : graph.get(at)) {
            if (visited[e.to]) continue;
            dfs(e.to);
        }

        f[at] = timeStamp++;
    }

    static int N;
    static int timeStamp = 1;
    static List<List<Edge>> graph;
    static boolean[] visited;
    static int[] d;
    static int[] f;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        N = Integer.parseInt(sc.next());
        graph = new ArrayList<>();
        for (int i = 0; i < N; ++i) graph.add(new ArrayList<>());
        visited = new boolean[N];
        d = new int[N];
        f = new int[N];

        for (int i = 0; i < N; ++i) {
            int id = Integer.parseInt(sc.next());
            int degree = Integer.parseInt(sc.next());
            for (int j = 0; j < degree; ++j) {
                int v = Integer.parseInt(sc.next());
                graph.get(i).add(new Edge(i, v-1));
            }
        }

        for (int i = 0; i < N; ++i) {
            dfs(i);
        }

        for (int i = 0; i < N; ++i) {
            System.out.println((i+1) + " " + d[i] + " " + f[i]);
        }
    }
}
