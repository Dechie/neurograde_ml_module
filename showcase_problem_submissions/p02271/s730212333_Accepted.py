n = int(input())
a = list(map(int, input().split()))
q = int(input())
m = list(map(int, input().split()))

def main():
    for number in m:
        if solve(0, number):
            print('yes')
        else :
            print('no')

def memolize(f):
    cache = {}

    def helper(x, y):
        if (x, y) not in cache:
            cache[(x, y)] = f(x, y)
        return cache[(x, y)]
    return helper

@memolize
def solve(i, m):
    if m == 0:
        return True

    if i >= n:
        return False

    res = solve(i + 1, m) or solve(i + 1, m - a[i])
    return res

if __name__ == '__main__':
    main()

