from collections import defaultdict

s = input()

def judge() :
    if len(s) == 3 :
        if s[0] == s[2] :
            return 'Second'
        else :
            return 'First'

    d = defaultdict(int)
    
    for s_ in s :
        d[s_] += 1
    
    if len(d) == 3 :
        one = []
        for k, v in d.items() :
            if v == 1 :
                one.append(k)
        
        if len(one) == 1 :
            for i in range(1, len(s) - 1) :
                if s[i] == one[0] :
                    if s[i-1] != s[i+1] :
                        return 'Fisrt'
                    else :
                        break
    
    if (len(s) + 1 if s[0] == s[-1] else 0) % 2 == 1 :
        return 'First'
    else :
        return 'Second'
       
print(judge())