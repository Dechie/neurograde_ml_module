def solve(A, m):
	if m==0:
		return True
	if len(A) == 0:
		return False
	if m < 0:
		return False
	else:
		return (solve(A[1:], m) or solve(A[1:], m-int(A[0])))


def main():
	n = int(input())
	A = input()
	A = A.split()
	q = int(input())
	ms= input()
	for m in ms.split():
		if solve(A, int(m)):
			print('yes')
		else:
			print('no')

if __name__ == '__main__':
	main()