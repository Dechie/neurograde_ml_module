import java.util.*;
public class Main {
    public static void main(String [] args){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int Edge[][] = new int[n][n];
        for(int i=0;i<n;i++){
            int v = sc.nextInt();
            int k= sc.nextInt();
            Arrays.fill(Edge[v-1],0);
            for(int j=0;j<k;j++){
                int num = sc.nextInt();
                Edge[v-1][num-1]=1;
            }
        }
        int [] First_Time = new int[n];
        int [] Last_Time = new int[n];
        int Check[] = new int[n];
        Arrays.fill(Check,0);
        int ts = 0;
        for(int i=0;i<n;i++) if(Check[i]==0) ts=DFS(Edge,First_Time,Last_Time,Check,ts,i);
        for(int i=0;i<n;i++) System.out.println(i+1+" "+First_Time[i]+" "+Last_Time[i]);
    }
    public static int DFS(int Edge[][],int FT[],int LT[],int Check[],int ts,int v){
        ts++;
        FT[v] = ts;
        Check[v]=1;
        for(int i=0;i<Edge[v].length;i++) if(Check[i]==0&&Edge[v][i]==1)ts=DFS(Edge,FT,LT,Check,ts,i);
        ts++;
        LT[v] = ts;
        return ts;
    }
}
