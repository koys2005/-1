'''라이브러리 소개
1. math
2. itertools
3. numpy
4. sys
5. matplotlib
6. sympy
'''

import math
import itertools as it
import numpy as np
import sys
import matplotlib.pyplot as plt
from sympy import Symbol, symbols, Eq, solve, nsolve, sin, diff, pprint

# print(dir(math))
# print(dir(it))
# print(help(print))
# sys.stdout = open('./ComputerScienceClass/output.txt', 'w', encoding='utf-8')
# print('Hello World!!!')

sharps = 50

##### math ###################################
print('\n#####', 'math     ', '#' * sharps)
print('pi\t\t:', math.pi)
print('e\t\t:', math.e)
print('sqrt\t\t:', math.sqrt(5))
print('atan2\t\t:', math.degrees(math.atan2(1,1)))
print('factorial\t:', math.factorial(10))
print('permutation\t:', math.perm(4,2))
print('combination\t:', math.comb(4,2))
print('gcd\t\t:', math.gcd(12,24,96))
print('lcm\t\t:', math.lcm(12,24,96))

##### itertools ###################################
print('\n\n#####', 'itertools', '#' * sharps)
print('조합\t\t:', list(it.combinations('ABCD',2)))
print('순열\t\t:', list(it.permutations('ABCD',2)))
print('중복조합\t:', list(it.combinations_with_replacement('ABCD',2)))
print('데카르트 곱\t:', list(it.product('ABCD','12')))

##### numpy ###################################
print('\n\n#####', 'numpy    ', '#' * sharps)
A = np.array([[1, 0, -1], [0, 2, 1]])
B = np.array([[1, 1], [-3, 1], [2, 0]])
print(A)
print(B)
C=A@B
D=np.linalg.inv(C)
print(C, end=' ')
print('\t행렬값 :', np.linalg.det(C))
print(D, '\t<-- 역행렬')
print(C@D, '\t<-- 항등행렬')

##### sympy ###################################
print('\n\n#####', 'sympy    ', '#' * sharps)
a, b, c = symbols('a, b, c')
x = Symbol('x')
expr = a * x**2 + b * x + c
print('근의 공식')
pprint(solve( Eq(expr, 0), x ))

expr=expr.subs( {a:x**2, b:-3*(x+1), c:3*x-4} )
print('수식\t\t:', expr)
print('전개\t\t:', expr.expand())
print('인수분해\t:', expr.factor())
print('근\t\t:', solve( Eq(expr ,0) ,x ))

# 수치적 해법 및 그래프
ans = solve(2 * sin(x) ** 2 + sin(x) - 1, x)
print('정해\t\t:', ans)
equation = 2 * x + 6 * sin(x) - 30
numAns = nsolve(equation, x, 18)
print('수치해법\t:', numAns)

##### 활용 ###################################
print('\n\n#####', '활용     ', '#' * sharps)
# 멱집합 구하기
s = list('ABC')
print('멱집합\t\t:', list( it.chain.from_iterable( it.combination(s, r) for r in range(len(s)+1) ) ))

# 소인수분해 사용자 모듈 생성 후 호출하기
from MyPyLib import prime_factoring
pf = prime_factoring( math.factorial(10) )
print('소인수분해\t:', eval(pf), ' = ', pf)

# 확률밀도함수 계산
from MyPyLib import solve14583
print('백준 14853번 문제')
print(solve14853(2, 1, 4, 3))
print(solve14853(8, 4, 16, 8))
print(solve14853(2, 0, 6, 1))
print(solve14853(2, 0, 2, 1))

# 그래프 활용
x = np.linspace(0, 2 * np.pi, 1024)
sin_x = np.sin(x)
diff_sin_x = (sin_x[1:] - sin_x[:-1]) / (2 * np.pi  / 1024)

plt.plot(x, sin_x, 'r')
plt.plot(x[:-1], diff_sin_x, 'b')

plt.xticks([0, np.pi / 2, np.pi, np.pi / 2 * 3, np.pi * 2])
plt.grid(linestyle='-', color='0.1', linewidth=0.5)
plt.show()