import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.util.ArrayList;
/**
 * 境界部分には気をつける(そこだけ別で処理するというのもあり)
 * 逆再生することで簡単になることもある
 */
public class Main {
	public int cnt = 0;
	public static final BigDecimal TWO = new BigDecimal(2);
	public static final BigDecimal THREE = new BigDecimal(3);

    public Main() {
    }

    public static void main(String[] args ) {
            Main main = new Main();
            main.exec();
    }

    public void exec() {
        BufferedReader bf = null;
        try{
            bf = new BufferedReader(new InputStreamReader(System.in));
            exec_body(bf);
        } catch (IOException e) {
            e.printStackTrace();
        }
        finally{
            if( bf != null ) {
                try {
                    bf.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public class Card {
    		String _character = null;
    		int _number = 0;

    		public Card( String chara, int number ) {
    			_character = chara;
    			_number = number;
    		}
    }

    public void exec_body( BufferedReader bf ) throws IOException {
    		String s = bf.readLine();
    		ArrayList<Card> cardList = new ArrayList<>(Integer.valueOf(s));
    		String line = bf.readLine();
    		while( line != null && line.length() > 0 ) {
    			String[] elements = line.split(" ");
    			Card card = new Card(elements[0],Integer.parseInt(elements[1]));
    			cardList.add(card);
    		}
    		bf.close();

       	Card[] mergeCards = cardList.toArray(new Card[cardList.size()]);
    		mergeSort(mergeCards, 0, mergeCards.length-1);
    		Card[] quickCards = cardList.toArray(new Card[cardList.size()]);
    		quickSort(quickCards, 0, quickCards.length -1);

    		if( isSameAry(mergeCards, quickCards)) {
    			System.out.println("Stable");
    		} else {
    			System.out.println("Not stable");
    		}

    		for( Card quickCard : quickCards ) {
    			System.out.println( quickCard._character + " " + quickCard._number );
    		}
    }

    public static boolean isSameAry( Card[] srcAry, Card[] dstAry ) {
    		if( srcAry.length != dstAry.length ) {
    			return false;
    		}

		for( int ii = 0; ii < srcAry.length; ii++ ) {
			if( srcAry[ii]._number != dstAry[ii]._number ) {
				return false;
			}
		}

		return true;
    }

    public void mergeSort( Card[] cards, int start, int last ) {
    		if( start >= last ) {
    			return;
    		}

    		int middle = ( start + last ) / 2;
    		mergeSort(cards,start,middle);
    		mergeSort(cards,middle+1,last);
    		merge(cards,start,last,middle);
    }

    public void merge( Card[] cards, int start, int last, int middle ) {
    		Card[] leftCards = new Card[middle-start+2];
    		for( int ii = 0; ii < middle-start+1; ii++ ) {
    			leftCards[ii] = cards[ii+start];
    		}
    		leftCards[middle-start+1] = new Card("D",Integer.MAX_VALUE);

    		Card[] rightCards = new Card[last-middle+1];
    		for( int ii = 0; ii < last-middle; ii++ ) {
    			rightCards[ii] = cards[ii+middle+1];
    		}
    		rightCards[last-middle] = new Card("D",Integer.MAX_VALUE);;

    		int leftIndex = 0;
    		int rightIndex = 0;
    		for( int ii = start; ii <= last; ii++ ) {
    			if( leftCards[leftIndex]._number <= rightCards[rightIndex]._number ) {
    				cards[ii] = leftCards[leftIndex];
    				leftIndex++;
    			} else {
    				cards[ii] = rightCards[rightIndex];
    				rightIndex++;
    			}
    		}
    }

    public void quickSort( Card[] cards, int start, int last ) {
    		if ( start < last ) {
    			int index = partition(cards, start, last);
    			quickSort(cards, start, index-1);
    			quickSort(cards, index + 1, last);
    		}
    }

    public int partition( Card[] cards, int start, int last ) {
    		Card lastCard = cards[last];
    		int insertIndex = start -1;
    		for( int ii = start; ii < last; ii++ ) {
    			Card card = cards[ii];
    			if( card._number <= lastCard._number) {
    				insertIndex++;
    				cards[ii] = cards[insertIndex];
    				cards[insertIndex] = card;
    			}
    		}
    		cards[last] = cards[insertIndex+1];
    		cards[insertIndex+1] = lastCard;
    		return insertIndex+1;
    }

    public static int[] toIList( String str ) {
        String[] elements = str.split(" ");
        int[] ret = new int [elements.length];
        for( int i = 0; i < elements.length; i++ ) {
            ret[i] = Integer.valueOf(elements[i]);
        }

        return ret;
    }

    public static void print( int[] ary ) {
    		if( ary == null || ary.length == 0 ) {
    			return;
    		}

    		StringBuilder bd = new StringBuilder();
    		for( int i = 0; i < ary.length; i++ ) {
    			if( i != 0 ) {
    				bd.append(" ");
    			}
    			bd.append(ary[i]);
    		}

    		System.out.println(bd.toString());
    }
}
