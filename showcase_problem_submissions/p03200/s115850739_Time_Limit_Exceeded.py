S = list(input())
count = 0

def black_and_white(S):
  global count
  for i in range(0,len(S)-1):
    if(S[i] == "B" and S[i+1] == "W"):
      count += 1
      tmp = S[i]
      S[i] = S[i+1]
      S[i+1] = tmp
      return True
  return False

while(1):
  if not black_and_white(S): break

print(count)

