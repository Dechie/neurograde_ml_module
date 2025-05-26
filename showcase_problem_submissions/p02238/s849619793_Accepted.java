import java.util.Arrays;

class Main {
    public static void main(String args[]) {
        java.util.Scanner sc = new java.util.Scanner(System.in);
        int n = Integer.parseInt(sc.next());
        int[][] adjacency = new int[n][n];

        for(int i=0; i<n; i++){
            int vIndex = Integer.parseInt(sc.next())-1;
            int nAdj = Integer.parseInt(sc.next());
            for(int j=0; j<nAdj; j++){
                int connectTo = Integer.parseInt(sc.next())-1;
                adjacency[vIndex][connectTo] = 1;
            }
        }

        DFS dfs = new DFS(adjacency);
        for(int i=0; i<dfs.getNp(); i++){
            dfs.execute(i);
        }
        
        dfs.printAnswer(); 
    }


    static class DFS{
        private final  int[][] adjacency;
        private final int[][] ans;
        private final Status[] status;
        private int time = 1;
        private final int np;

        public int getNp(){
            return this.np;
        }

        public DFS(int[][] adjacency){
            this.adjacency = adjacency;
            this.np = adjacency.length;
            this.ans = new int[adjacency.length][3];
            for(int i=0; i<np; i++){
                this.ans[i][0] = i+1;
            }
            Status[] status = new Status[np];
            Arrays.fill(status, Status.NOT_YET);
            this.status = status;
        }

        private void searchNotYet(int vi){
            for(int i=0; i<np; i++){
                if( adjacency[vi][i] == 1 && i != vi && this.status[i] == Status.NOT_YET){
                    int iNotYet = i;
                    execute(iNotYet);
                }
            }
        }

        public void execute(int vi){
            if(this.status[vi] != Status.NOT_YET) return;
            ans[vi][1] = time++;
            status[vi] = Status.CURRENT;
            searchNotYet(vi);
            ans[vi][2] = time++;
        }
        public void printAnswer(){
            for(int i=0; i<np; i++){
                for(int j=0; j<3; j++) {
                    if(j== 2)   System.out.println(ans[i][j]);
                    else System.out.print(ans[i][j]+" ");
                }
            }
        }
    }

    static enum Status{
        DONE,
        CURRENT,
        NOT_YET
    }
}

