r = 1000001
s = int(r**0.5)
p = [1]*r
p[0] = p[1] = 0
for i in range(s):
	if p[i]:
		p[2*i::i] = [0 for j in range(2*i,r,i)]

while 1:
	a,d,n = map(int,raw_input().split())
	if n == 0: break
	c = 0
	for i in range(a,r,d):
		if p[i]: 
			c += 1
		if c == n:
			print i
			break
	