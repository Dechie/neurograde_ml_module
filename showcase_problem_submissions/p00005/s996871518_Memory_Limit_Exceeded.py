while True:
	try:
		a,b = map(int, sorted(raw_input().split()))
		# GCD
		for i in range(a, 1, -1):
			if a%i == 0 and b%i == 0:
				GCD = i
				break
		# LCM
		for i in range(1, a+1):
			if (b * i) % a == 0:
				LCM = b * i
				break
		print " ".join(map(str,[GCD, LCM]))[:-1]
	except EOFError:
		break