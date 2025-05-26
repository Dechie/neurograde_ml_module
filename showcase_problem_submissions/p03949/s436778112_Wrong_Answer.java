import java.util.ArrayList;
import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.Scanner;

public class Main {
	@SuppressWarnings("unchecked")
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		int N = sc.nextInt();
		ArrayList<Edge>[] g = new ArrayList[N + 1];
		for (int i = 0; i < N + 1; ++i) {
			g[i] = new ArrayList<>();
		}
		for (int i = 0; i < N - 1; ++i) {
			int a = sc.nextInt() - 1;
			int b = sc.nextInt() - 1;
			g[a].add(new Edge(a, b, 1));
			g[b].add(new Edge(b, a, 1));
		}

		int K = sc.nextInt();
		int[] V = new int[K];
		int[] P = new int[K];
		for (int i = 0; i < K; ++i) {
			V[i] = sc.nextInt() - 1;
			P[i] = sc.nextInt();
			g[N].add(new Edge(N, V[i], P[i]));
		}

		class State implements Comparable<State> {
			int cur;
			long dis;

			public State(int cur, long dis) {
				this.dis = dis;
				this.cur = cur;
			}

			public int compareTo(State arg0) {
				return Long.compare(dis, arg0.dis);
			};
		}

		PriorityQueue<State> pq = new PriorityQueue<>();

		long[] d = new long[N+1];
		Arrays.fill(d, Long.MAX_VALUE);
		d[N] = 0;
		pq.add(new State(N, 0));

		while (!pq.isEmpty()) {
			State s = pq.poll();
			if (d[s.cur] < s.dis)
				continue;
			else
				d[s.cur] = s.dis;

			for (Edge e : g[s.cur]) {
				if (d[e.dst] > d[e.src] + e.cost) {
					d[e.dst] = d[e.src] + e.cost;
					pq.add(new State(e.dst, d[e.dst]));
				}
			}
		}

		for (int i = 0; i < N; ++i) {
			for (Edge e : g[i]) {
				if (Math.abs(d[e.src] - d[e.dst]) != 1) {
					System.out.println("No");
					return;
				}
			}
		}

		System.out.println("Yes");
		for (int i = 0; i < N; ++i) {
			System.out.println(d[i]);
		}
	}

	static class Edge {
		int src;
		int dst;
		int cost;

		public Edge(int src, int dst, int cost) {
			this.src = src;
			this.dst = dst;
			this.cost = cost;
		}
	}
}
