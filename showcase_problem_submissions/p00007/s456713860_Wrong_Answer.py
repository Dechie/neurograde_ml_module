import sys

def main():
    n = int(input().rstrip())
    r = 1.05
    digit = -4
    a = 100000
    
    print(int(round(a*(r**n), digit)))
    
if __name__ == '__main__':
    main()