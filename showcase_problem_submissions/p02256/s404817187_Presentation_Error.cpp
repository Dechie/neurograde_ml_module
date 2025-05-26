#include<stdio.h>

int main(void)
{

	int x, y, d;

	scanf("%d", &x);
	scanf("%d", &y);

	d = x % y;

	while (d != 0) {

		x = y;
		y = d;
		d = x % y;
	}
	printf("%d", y);
	

	return 0;
}
