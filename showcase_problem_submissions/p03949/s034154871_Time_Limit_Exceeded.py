#設定
import sys
input = sys.stdin.buffer.readline
sys.setrecursionlimit(10**7)

from collections import defaultdict
import queue

INF = float("inf")

#入力受け取り
def getlist():
	return list(map(int, input().split()))

class Graph(object):
	def __init__(self):
		self.graph = defaultdict(list)

	def __len__(self):
		return len(self.graph)

	def add_edge(self, a, b):
		self.graph[a].append(b)

	def get_nodes(self):
		return self.graph.keys()

class BFS(object):
	def __init__(self, graph, s, N):
		self.g = graph.graph
		self.Q = queue.Queue()
		self.Q.put(s)
		self.dist = [INF] * N
		self.dist[s] = 0
		self.visit = ["No"] * N
		self.visit[s] = "Yes"
		while not self.Q.empty():
			v = self.Q.get()
			for i in self.g[v]:
				if self.visit[i] == "No":
					self.dist[i] = self.dist[v] + 1
					self.Q.put(i)
					self.visit[i] = "Yes"

def DFS(G, MAX, MIN, visit, node):
	for i in G.graph[node]:
		if visit[i] != "Yes":
			visit[i] = "Yes"
			DFS(G, MAX, MIN, visit, i)
			MAX[node] = min(MAX[node], MAX[i] + 1)
			MIN[node] = max(MIN[node], MIN[i] - 1)

class BFS2(object):
	def __init__(self, graph, s, N, MAX, MIN):
		self.g = graph.graph
		self.Q = queue.Queue()
		self.Q.put(s)
		self.visit = ["No"] * N
		self.visit[s] = "Yes"
		while not self.Q.empty():
			v = self.Q.get()
			for i in self.g[v]:
				if self.visit[i] == "No":
					self.visit[i] = "Yes"
					self.Q.put(i)
					MAX[i] = min(MAX[i], MAX[v] + 1)
					MIN[i] = max(MIN[i], MIN[v] - 1)

#処理内容
def main():
	#入力、前処理
	N = int(input())
	G = Graph()
	for i in range(N - 1):
		a, b = getlist()
		a -= 1; b -= 1
		G.add_edge(a, b); G.add_edge(b, a)
	K = int(input())
	MAX = [INF] * N; MIN = [-INF] * N
	for i in range(K):
		V, P = getlist()
		V -= 1
		MAX[V] = P
		MIN[V] = P

	#BFS 偶奇判定
	BF = BFS(G, V, N)
	dist = BF.dist
	odd_even = [0, 0]
	for i in range(N):
		if MAX[i] != INF:
			if MAX[i] & 1 == dist[i] & 1:
				odd_even[0] += 1
			else:
				odd_even[1] += 1
	
	#例外処理
	if odd_even[0] * odd_even[1] > 0:
		print("No")
		return

	#DFS
	visit = ["No"] * N
	visit[V] = "Yes"
	DFS(G, MAX, MIN, visit, V)
	#BFS
	BF2 = BFS2(G, V, N, MAX, MIN)

	#判定
	judge = "Yes"
	for i in range(N):
		if MAX[i] < MIN[i]:
			judge = "No"
			break
	if judge == "No":
		print(judge)
		return

	print("Yes")
	for i in MAX:
		print(i)

if __name__ == '__main__':
	main()