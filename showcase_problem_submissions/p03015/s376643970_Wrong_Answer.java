import java.io.InputStream;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.Scanner;

import javax.naming.ldap.HasControls;

import com.sun.org.apache.bcel.internal.generic.LLOAD;
import com.sun.org.apache.xpath.internal.operations.Mod;


public class Main {
  public static void main(String args[])
  { Scanner cin=new Scanner(System.in);
  BigInteger sum1=cin.nextBigInteger();
  String s=sum1.toString(2);
  char s1[]=s.toCharArray();
  int sum=s.length();
  int a=0;
  for(int i=0;i<sum;i++)
	   if(s1[i]=='1')a++;
   BigInteger x=new BigInteger("1");
   BigInteger y=BigInteger.valueOf(2);
   BigInteger xx=BigInteger.valueOf(1000000007);
  System.out.println(y.pow(a).add(x).remainder(xx));
    // BigInteger sum2=new BigInteger("3");
  //System.out.println(sum2.nextProbablePrime() );
     }
  
}
