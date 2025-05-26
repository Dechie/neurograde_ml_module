import sys
input=sys.stdin.readline

def main():
    S = input().strip()
    w = 0
    n = 0
    for i,c in enumerate(S):
        if c == "W":
            n += i-w
            w += 1
    print(n)

if __name__ == '__main__':
    main()