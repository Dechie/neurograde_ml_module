/*16D8101024H Muramatsu Masaki 村松 将希 mentuyu C++*/

#include<iostream>

int gcd(int a,int b){
  int r,tmp;
  if(a<b){
    tmp = a;
    a = b;
    b = tmp;
  }

  r = a % b;

  while(r!=0){
    a = b;
    b = r;
    r = a % b;
  }
  return b;
}

int main(){
  int x,y,z;
  std::cin >> x >> y;
  z = gcd(x,y);
  std::cout<<z;
  return 0;
}

