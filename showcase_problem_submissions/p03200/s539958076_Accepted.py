def main():
    s = input()
    length = len(s)-1
    count = s.count('B')
    init = sum([i for i in range(len(s)) if s[i] == "B"])
    processed = sum(range(length-count+1, length+1))

    print(processed-init)


if __name__ == '__main__':
    main()
