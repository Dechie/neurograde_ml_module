import java.math.BigInteger;
import java.util.*;
import java.util.function.BinaryOperator;
import java.util.regex.Matcher;
import java.util.stream.Collectors;

public class Main {
  private static BigInteger MOD = new BigInteger("998244353");

  public static void main(String[] args){
    Scanner sc = new Scanner(System.in);
    BigInteger a = new BigInteger(sc.next());
    BigInteger b = new BigInteger(sc.next());
    BigInteger c = new BigInteger(sc.next());
    BigInteger d = new BigInteger(sc.next());

    BigInteger tmp = culc(a, b, c, d);
    System.out.println(tmp);
  }

  private static Map<BigInteger, Map<BigInteger, BigInteger>> POW = new HashMap<>();

  static BigInteger pow(BigInteger i, BigInteger k) {
    return POW.computeIfAbsent(i, key -> new HashMap<>()).computeIfAbsent(k, key -> {
      if (k.compareTo(BigInteger.ZERO) == 0) {
        return BigInteger.ONE;
      }
      return i.multiply(pow(i, k.subtract(BigInteger.ONE))).remainder(MOD);
    });
  }

  private static Map<BigInteger, Map<BigInteger, BigInteger>> MAP = new HashMap<>();

  static BigInteger culc(BigInteger a, BigInteger b, BigInteger c, BigInteger d) {
    Map<BigInteger, BigInteger> bigIntegerBigIntegerMap = MAP.computeIfAbsent(c, k -> new HashMap<>());
    return bigIntegerBigIntegerMap.computeIfAbsent(d, k -> {
      if (a.compareTo(c) == 0) {
        return pow(a, d.subtract(b));
      }
      BigInteger minus = c.subtract(BigInteger.ONE);
      BigInteger result = b.multiply(culc(a, b, minus, b)).remainder(MOD).multiply(pow(c, d.subtract(b))).remainder(MOD);
      for (BigInteger i = b.add(BigInteger.ONE); i.compareTo(d) <= 0; i = i.add(BigInteger.ONE)) {
        result = result.add(culc(a, b, minus, i).multiply(pow(c, d.subtract(i))).remainder(MOD)).remainder(MOD);
      }
      return result;
    });
  }
}
