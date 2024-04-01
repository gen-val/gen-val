from   itertools  import permutations, combinations, product, chain
from   pprint     import pprint as pp
from   fractions  import Fraction as F
import random, ast, re
import sys
import pandas as pd
 
if sys.version_info[0] < 3:
    input = raw_input
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest
 
 
def choose4():
    'four random digits >0 as characters'
    return [str(random.randint(1,9)) for i in range(4)]

 
def check(answer, digits):
    allowed = set('() +-*/\t'+''.join(digits))
    ok = all(ch in allowed for ch in answer) and \
         all(digits.count(dig) == answer.count(dig) for dig in set(digits)) \
         and not re.search('\d\d', answer)
    if ok:
        try:
            ast.parse(answer)
        except:
            ok = False
    return ok
 
def solve(digits):
    digilen = len(digits)
    # length of an exp without brackets 
    exprlen = 2 * digilen - 1
    # permute all the digits
    digiperm = sorted(set(permutations(digits)))
    # All the possible operator combinations
    opcomb   = list(product('+-*/', repeat=digilen-1))
    # All the bracket insertion points:
    brackets = ( [()] + [(x,y)
                         for x in range(0, exprlen, 2)
                         for y in range(x+4, exprlen+2, 2)
                         if (x,y) != (0,exprlen+1)]
                 + [(0, 3+1, 4+2, 7+3)] ) # double brackets case
    for d in digiperm:
        for ops in opcomb:
            if '/' in ops:
                d2 = [('F(%s)' % i) for i in d] # Use Fractions for accuracy
            else:
                d2 = d
            ex = list(chain.from_iterable(zip_longest(d2, ops, fillvalue='')))
            for b in brackets:
                exp = ex[::]
                for insertpoint, bracket in zip(b, '()'*(len(b)//2)):
                    exp.insert(insertpoint, bracket)
                txt = ''.join(exp)
                try:
                    num = eval(txt)
                except ZeroDivisionError:
                    continue
                if num == 24:
                    if '/' in ops:
                        exp = [ (term if not term.startswith('F(') else term[2:-1])
                               for term in exp ]
                    ans = ' '.join(exp).rstrip()
                    try:
                        num = eval(ans)
                    except ZeroDivisionError:
                        print(txt, ans)
                        continue
                    if round(num, 1) != 24:
                        print(txt, ans)
                        continue
                    return ans
    return None

def main():
    fn = "2024_02_13_24game_n200_gpt-4-1106-preview.csv"
    df = pd.read_csv(f"../../24game/results/{fn}")

    sols = []
    num_solvable = 0
    for _, row in df.iterrows():
        digits = row['Question'].split()
        sol = solve(digits)
        try:
            if eval(sol) == 24:
                sols.append(sol + " = 24")
                num_solvable += 1
            else:
                sols.append(None)
                print(sol)
        except:
            sols.append(None)
            print(sol)
            
    print(f"% solvable: {num_solvable/len(df)*100}")
    df['Solution'] = sols
    df.to_csv(f"../../results/{fn}", index=False)

if __name__ == '__main__':
    main()