//package Qe;

import java.io.*;
import java.util.*;
import java.util.Scanner;


public class Main {
    private static Scanner sc = new Scanner(System.in);
    private static int readInt() {return sc.nextInt();}
    private static String readWord() { return sc.next(); } 

    public static void main(String[] args) {
        int N = readInt();
        String S = readWord();

        SegTree[] sts = new SegTree[26];
        for (int i = 0; i<sts.length; i++) {
            sts[i] = new SegTree(N);
        }

        for (int i=0; i<N; i++) {
            sts[S.charAt(i) - 'a'].add(i, 1);
        }

        int Q = readInt();

        char[] ca = S.toCharArray();
        for (int q=0; q<Q; q++) {

            int type = readInt();
            String arg1 = readWord();
            String arg2 = readWord();

            switch (type) {
                case 1:
                    int I = Integer.parseInt(arg1) - 1;
                    char C = arg2.charAt(0);
                    if (ca[I] != C) {
                        sts[ca[I] - 'a'].add(I, -1);
                        sts[C - 'a'].add(I, 1);
                        ca[I] = C;
                    }
                    break;
                case 2:
                    int L = Integer.parseInt(arg1)-1;
                    int R = Integer.parseInt(arg2);
                    int ans = 0;
                    for (int i = 0; i<sts.length; i++) {
                        if (sts[i].sum(L, R) > 0) ans++;
                    }
                    System.out.println(ans);
                    break;
            }
        }
    }

    static class SegTree {
        private long[] sg;
        private long[] sg2;
        private int sz;
        private int N;

        public SegTree(int N) {
            this.N = N;
            int sz = 1;
            while (sz < N) {
                sz <<= 1;
            }
            this.sz = sz;
            sg = new long[sz << 1];
            sg2 = new long[sz << 1];
        }

        void add(int i, int j, long X, int v, int l, int r) {
            if (r <= i || j <= l) {
                return;
            }
            if (i <= l && r <= j) {
                sg2[v] += X;
            } else {
                sg[v] += (Math.min(j, r) - Math.max(i, l)) * X;
                int m = (l + r) / 2;
                add(i, j, X, v * 2 + 1, l, m);
                add(i, j, X, v * 2 + 2, m, r);
            }
        }

        void add(int i, long X) {
            add(i, i+1, X, 0, 0, this.sz);
        }

        long sum(int i, int j, int v, int l, int r) {
            if (r <= i || j <= l) {
                return 0L;
            } else if (i <= l && r <= j) {
                return sg[v] + (r - l) * sg2[v];
            } else {
                long base = (Math.min(j, r) - Math.max(i, l)) * sg2[v];
                int m = (l + r) / 2;
                long vl = sum(i, j, v * 2 + 1, l, m);
                long vr = sum(i, j, v * 2 + 2, m, r);
                return base + vl + vr;
            }
        }

        long sum(int i, int j) {
            return sum(i, j, 0, 0, this.sz);
        }
    }
}