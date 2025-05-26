#include <iostream>
#include <math.h>

using namespace std;


int main()
{
	int i,n;
	cin >> n;
	
	int gaku = 100;

	for( i = 0; i < n; i++ ) {
		gaku = ceil(gaku * 1.05);
	}
	
	cout << gaku * 1000;

	return 0;
}