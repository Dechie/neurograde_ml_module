import java.util.*;
import java.io.*;

class Main{
    public static void main(String[] args){
        FastScanner fsc=new FastScanner();
        int n=fsc.nextInt();
        long m=fsc.nextLong();
        long[] a=new long[n];
        for(int i=0;i<n;i++){
            a[i]=fsc.nextLong();
        }
        revSort(a);
        fsc.close();
        long[] accum=new long[n+1];
        for(int i=1;i<=n;i++){
            accum[i]=accum[i-1]+a[i-1];
        }
        int pass=0;
        int fail=200001;
        while(fail-pass>1){
            int mid=pass+(fail-pass)/2;
            if(pass(mid, a, m)) pass=mid;
            else fail=mid;
        }

        long ans=0;
        long count=0;
        for(int i=0;i<n;i++){
            int p=-1;
            int f=n;
            while(f-p>1){
                int mid=p+(f-p)/2;
                if(a[i]+a[mid]>pass) p=mid;
                else f=mid;
            }
            count+=p+1;
            ans+=accum[p+1]+a[i]*(p+1);
        }
        ans+=pass*(m-count);
        System.out.println(ans);
    }
    static boolean pass(int x, long[] a, long m){
        int n=a.length;
        long count=0;
        for(int i=0;i<n;i++){
            int pass=-1;
            int fail=n;
            while(fail-pass>1){
                int mid=pass+(fail-pass)/2;
                if(a[i]+a[mid]>=x) pass=mid;
                else fail=mid;
            }
            count+=pass+1;
            if(count>=m) return true;
        }
        return false;
    }
    static void revSort(long[] a){
        Arrays.sort(a);
        int l=a.length;
        long tmp;
        for(int i=0;i<l/2;i++){
            tmp=a[i];
            a[i]=a[l-i-1];
            a[l-i-1]=tmp;
        }
    }
    static class FastScanner {
        private final InputStream in = System.in;
        private final byte[] buffer = new byte[1024];
        private int ptr = 0;
        private int buflen = 0;
        private boolean hasNextByte(){
            if(ptr<buflen) return true;
            else{
                ptr = 0;
                try{
                    buflen = in.read(buffer);
                }
                catch(IOException e){
                    e.printStackTrace();
                }
                if(buflen<=0) return false;
            }
        return true;
        }
        private int readByte() { if (hasNextByte()) return buffer[ptr++]; else return -1;}
        private static boolean isPrintableChar(int c) { return 33 <= c && c <= 126;}
        public boolean hasNext() { while(hasNextByte() && !isPrintableChar(buffer[ptr])) ptr++; return hasNextByte();}
        public String next() {
            if (!hasNext()) throw new NoSuchElementException();
            StringBuilder sb = new StringBuilder();
            int b = readByte();
            while(isPrintableChar(b)) {
                sb.appendCodePoint(b);
                b = readByte();
            }
            return sb.toString();
        }
        public long nextLong() {
            if (!hasNext()) throw new NoSuchElementException();
            long n = 0;
            boolean minus = false;
            int b = readByte();
            if (b == '-') {
                minus = true;
                b = readByte();
            }
            if (b < '0' || '9' < b) throw new NumberFormatException();
            while(true){
                if ('0' <= b && b <= '9') {
                    n *= 10;
                    n += b - '0';
                }
                else if(b == -1 || !isPrintableChar(b)) return minus ? -n : n;
                else throw new NumberFormatException();
                b = readByte();
            }
        }
        public int nextInt() {
            long nl = nextLong();
            if (nl < Integer.MIN_VALUE || nl > Integer.MAX_VALUE) throw new NumberFormatException();
            return (int) nl;
        }
        public double nextDouble() { return Double.parseDouble(next());}
        public void close(){
            try{in.close();}
            catch(IOException e){e.printStackTrace();}
        }
    }
}
