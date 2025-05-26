while True:
    try:
        print(int(round(100000 * (1.05 ** int(input())), -4)))
    except EOFError:
        break

