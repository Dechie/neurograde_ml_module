#include<stdio.h>

void depthfs(void);

void visit(int);

#define MAX 109

#define WHITE 0
#define GRAY 1
#define BLACK 2

int graph[MAX][MAX];

int n;
int time;
int d[MAX],f[MAX];
int color[MAX];

int main(){
  int num,i;
  int u,j,v;

  scanf("%d",&n);

  for(i=1;i<=n;i++){
    for(j=1;j<=n;j++){    
      graph[i][j] = 0;
    }
  }

  for(i=1;i<=n;i++){
    scanf("%d%d",&u,&num);
    for(j=0;j<num;j++){
      scanf("%d",&v);
      graph[u][v] = 1;
    }
  }

  depthfs();

  for(i=1;i<=n;i++){
    printf("%d %d %d\n",i,d[i],f[i]);
  }

  
  return 0;

}

void depthfs(void){
  int i;
  //for each vertex u in V
  for(i=1;i<=n;i++){
  color[i] = WHITE;
  d[i] = 0;
  f[i] = 0;
  }

  time= 0;

  //for each vertex u in V
  for(i=1;i<=n;i++){
  if(color[i] == WHITE) visit(i);
  }

}

void visit(int x){
  int i;

  color[x] = GRAY;//white vertex u has just been discovered

  d[x] = ++time;

  //for each v in adj[u] //explore edge(u,v)

  for(i=1;i<=n;i++){
    if(graph[x][i] == 1){
      if(color[i] == WHITE) visit(i);
    }
  }

  color[x] = BLACK; //blacken u; it is finished

  f[x] = ++time;
  
}

