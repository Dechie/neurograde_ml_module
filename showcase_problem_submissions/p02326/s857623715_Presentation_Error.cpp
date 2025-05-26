#include<iostream>
#include<stdio.h>
#include<algorithm>
#include<math.h>

using namespace std;

int H, W, C[1401][1401],Cup[1401][1401],Cleft[1401][1401],Cslash[1401][1401],result;

int main()
{
	for (int i = 0; i < 1401; i++)
	{
		for (int j = 0; j < 1401; j++)
		{
			C[i][j] = -1;
			Cup[i][j] = 0;
			Cleft[i][j] = 0;
			Cslash[i][j] = 0;
		}
	}

	scanf("%d", &H);
	scanf("%d", &W);

	for (int i = 1; i < H+1; i++)
	{
		for (int j = 1; j < W+1; j++)
		{
			scanf("%d", &C[i][j]);
		}
	}

	for (int i = 1; i < H+1; i++)
	{
		for (int j = 1; j < W+1; j++)
		{
			if (C[i][j] == 0)
			{
				Cup[i][j] = 1 + Cup[i - 1][j];
				Cleft[i][j] = 1 + Cleft[i][j - 1];
				Cslash[i][j] = 1 + Cslash[i - 1][j - 1];
				result = max(result, min(Cup[i][j], min(Cleft[i][j], Cslash[i][j])));
			}
		}
	}

	result *= result;

	printf("%d", result);

	return 0;
}