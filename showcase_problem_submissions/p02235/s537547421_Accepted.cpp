#include <iostream>
#include <vector>
using namespace  std;

void fnInput(vector<string>& rvsStrX, vector<string>& rvsStrY)
{
  int nMaxSiz;
  cin >> nMaxSiz;
  rvsStrX.resize(nMaxSiz);
  rvsStrY.resize(nMaxSiz);

  for (int i = 0; i < nMaxSiz; ++i)
  {
    cin >> rvsStrX[i] >> rvsStrY[i];
    rvsStrX[i] = ' ' + rvsStrX[i];
    rvsStrY[i] = ' ' + rvsStrY[i];
  }
}

int fnLngCmnSubseq(const vector<string>& cnrvsStrX, const vector<string>& cnrvsStrY)
{
  for (int i = 0; i < cnrvsStrX.size(); ++i)
  {
    const int nMAXNX = cnrvsStrX[i].size();
    const int nMAXNY = cnrvsStrY[i].size();
    vector<vector<int>> vvnLcsDP(nMAXNX, vector<int>(nMAXNY, 0));

    for (int nx = 1; nx < nMAXNX; ++nx)
      for (int ny = 1; ny < nMAXNY; ++ny)
        if (cnrvsStrX[i][nx] == cnrvsStrY[i][ny])
          vvnLcsDP[nx][ny] = vvnLcsDP[nx - 1][ny - 1] + 1;
        else
          vvnLcsDP[nx][ny] = max(vvnLcsDP[nx - 1][ny], vvnLcsDP[nx][ny - 1]);

    cout << vvnLcsDP[nMAXNX - 1][nMAXNY - 1] << endl;
  }
}

int main()
{
  cin.tie(0);
  ios::sync_with_stdio(false);
  vector<string> vsStrX, vsStrY;

  fnInput(vsStrX, vsStrY);
  fnLngCmnSubseq(vsStrX, vsStrY);
  return 0;
}