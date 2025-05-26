while True:
	try:
		n = int(raw_input())
		l = int(100000*pow(1.05, n))
		if l%10000 > 0:
			l -= l%10000
			l += 10000
		print l
	except EOFError:
		break