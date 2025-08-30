"""
λ-Calculus + Combinators + Church Arithmetic (Python)

A comprehensive, runnable reference for λ-calculus fundamentals including:
* Core combinators (I, K, KI, C, S, B, W, T, M)
* Church booleans and logic (NOT, AND, OR, equality)
* DeMorgan verification
* Church numerals (0, 1, 2, 3, …) + iteration helpers
* Arithmetic: successor, addition, multiplication, exponentiation, predecessor
* Pairs (for predecessor implementation)
* Predicates: isZero, leq, eq
* Translation of !x == y || (a && z) into λ-calculus
"""

 # YouTube videos

 # Lambda Calculus - Fundamentals of Lambda Calculus & Functional Programming in JavaScript
 # [https://www.youtube.com/watch?v=3VQ382QG-y4](https://www.youtube.com/watch?v=3VQ382QG-y4)

 # A Flock of Functions; Combinators, Lambda Calculus, & Church Encodings in JS - Part II
 # [https://www.youtube.com/watch?v=pAnLQ9jwN-E](https://www.youtube.com/watch?v=pAnLQ9jwN-E)

 # Online IDE
 # [https://www.programiz.com/javascript/online-compiler/](https://www.programiz.com/javascript/online-compiler/)

from typing import Callable, Any

# === HELPERS ===

# Convert Church numeral to Python int
def jsnum(n: Callable[[Callable[[int], int]], Callable[[int], int]]) -> int:
    return n(lambda x: x + 1)(0)

# Convert Church boolean to string
def show_bool(p: Callable[[str], Callable[[str], str]]) -> str:
    return p("TRUE")("FALSE")

# === 1) CORE COMBINATORS ===

# Identity (I) — λa.a
I: Callable[[Any], Any] = lambda a: a

# Kestrel (K) — λa.λb.a (TRUE)
K: Callable[[Any], Callable[[Any], Any]] = lambda a: lambda b: a

# Kite (KI) — λa.λb.b (FALSE)
KI: Callable[[Any], Callable[[Any], Any]] = lambda a: lambda b: b

# Cardinal (C) — λf.λa.λb.f b a (flip)
C: Callable[[Callable], Callable[[Any], Callable[[Any], Any]]] = \
    lambda f: lambda a: lambda b: f(b)(a)

# Starling (S) — λf.λg.λx.f x (g x)
S: Callable[[Callable], Callable[[Callable], Callable[[Any], Any]]] = \
    lambda f: lambda g: lambda x: f(x)(g(x))

# Bluebird (B) — λf.λg.λx.f (g x) (compose)
B: Callable[[Callable], Callable[[Callable], Callable[[Any], Any]]] = \
    lambda f: lambda g: lambda x: f(g(x))

# Warbler (W) — λf.λx.f x x
W: Callable[[Callable], Callable[[Any], Any]] = \
    lambda f: lambda x: f(x)(x)

# Thrush (T) — λa.λf.f a
T: Callable[[Any], Callable[[Callable], Any]] = \
    lambda a: lambda f: f(a)

# Mockingbird (M) — λf.f f (self-application)
M: Callable[[Callable], Any] = lambda f: f(f)

# === 2) CHURCH BOOLEANS ===

# TRUE and FALSE
TRUE = K
FALSE = KI

# NOT — λp.p FALSE TRUE
NOT: Callable[[Callable], Callable] = lambda p: p(FALSE)(TRUE)

# AND — λp.λq.p q FALSE
AND: Callable[[Callable], Callable[[Callable], Callable]] = \
    lambda p: lambda q: p(q)(FALSE)

# AND (alternative) — λp.λq.p q p
AND2: Callable[[Callable], Callable[[Callable], Callable]] = \
    lambda p: lambda q: p(q)(p)

# OR — λp.λq.p TRUE q
OR: Callable[[Callable], Callable[[Callable], Callable]] = \
    lambda p: lambda q: p(TRUE)(q)

# OR (alternative) — λp.λq.p p q
OR2: Callable[[Callable], Callable[[Callable], Callable]] = \
    lambda p: lambda q: p(p)(q)

# Boolean equality — λp.λq.p q (NOT q)
BEQ: Callable[[Callable], Callable[[Callable], Callable]] = \
    lambda p: lambda q: p(q)(NOT(q))

# === 3) CHURCH NUMERALS ===

# Zero — λf.λx.x
ZERO: Callable[[Callable], Callable[[Any], Any]] = lambda f: lambda x: x

# Successor — λn.λf.λx.f (n f x)
SUCC: Callable[[Callable], Callable[[Callable], Callable[[Any], Any]]] = \
    lambda n: lambda f: lambda x: f(n(f)(x))

# Build numerals
ONE = SUCC(ZERO)
TWO = SUCC(ONE)
THREE = SUCC(TWO)
FOUR = SUCC(THREE)
FIVE = SUCC(FOUR)

# Iteration helpers
once: Callable[[Callable], Callable[[Any], Any]] = lambda f: lambda a: f(a)
twice: Callable[[Callable], Callable[[Any], Any]] = lambda f: lambda a: f(f(a))
thrice: Callable[[Callable], Callable[[Any], Any]] = lambda f: lambda a: f(f(f(a)))
fourfold: Callable[[Callable], Callable[[Any], Any]] = lambda f: lambda a: f(f(f(f(a))))
fivefold: Callable[[Callable], Callable[[Any], Any]] = lambda f: lambda a: f(f(f(f(f(a)))))

# === 4) PAIRS (for predecessor) ===

# Church pair — λa.λb.λf.f a b
PAIR: Callable[[Any], Callable[[Any], Callable[[Callable], Any]]] = \
    lambda a: lambda b: lambda f: f(a)(b)

FST: Callable[[Callable], Any] = lambda p: p(K)
SND: Callable[[Callable], Any] = lambda p: p(KI)

# === 5) ARITHMETIC ===

# Addition — λm.λn.λf.λx.m f (n f x)
PLUS: Callable[[Callable], Callable[[Callable], Callable[[Callable], Callable[[Any], Any]]]] = \
    lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x))

# Multiplication — λm.λn.λf.m (n f)
MULT: Callable[[Callable], Callable[[Callable], Callable[[Callable], Callable[[Any], Any]]]] = \
    lambda m: lambda n: lambda f: m(n(f))

# Exponentiation — λm.λn.n m
POW: Callable[[Callable], Callable[[Callable], Callable]] = \
    lambda m: lambda n: n(m)

# Predecessor (via pairs)
STEP: Callable[[Callable], Callable] = \
    lambda p: PAIR(SND(p))(SUCC(SND(p)))

PRED: Callable[[Callable], Callable] = \
    lambda n: FST(n(STEP)(PAIR(ZERO)(ZERO)))

# Subtraction — λm.λn.n PRED m
SUB: Callable[[Callable], Callable[[Callable], Callable]] = \
    lambda m: lambda n: n(PRED)(m)

# === 6) PREDICATES ===

# Is zero — λn.n (λ_.FALSE) TRUE
ISZERO: Callable[[Callable], Callable] = \
    lambda n: n(lambda _: FALSE)(TRUE)

# Less than or equal — λm.λn.ISZERO (SUB m n)
LEQ: Callable[[Callable], Callable[[Callable], Callable]] = \
    lambda m: lambda n: ISZERO(SUB(m)(n))

# Numeric equality — λm.λn.AND (LEQ m n) (LEQ n m)
EQN: Callable[[Callable], Callable[[Callable], Callable]] = \
    lambda m: lambda n: AND(LEQ(m)(n))(LEQ(n)(m))

# === 7) EXPRESSION TRANSLATION ===

# !x == y || (a && z) becomes:
def EXPR(x, y, a, z):
    return OR(BEQ(NOT(x))(y))(AND(a)(z))

# === DEMONSTRATIONS ===

if __name__ == "__main__":
    print("=== COMBINATORS ===")
    print(I(42))                    # 42
    print(K(1)(2))                  # 1
    print(C(K)(1)(2))              # 2
    print(M(I))                     # <function>
    
    print("\n=== BOOLEANS ===")
    print(show_bool(TRUE))          # TRUE
    print(show_bool(FALSE))         # FALSE
    print(show_bool(NOT(TRUE)))     # FALSE
    print(show_bool(AND(TRUE)(FALSE)))  # FALSE
    print(show_bool(OR(FALSE)(TRUE)))   # TRUE
    
    print("\n=== DEMORGAN VERIFICATION ===")
    # ¬(p ∧ q) ≡ (¬p) ∨ (¬q)
    def de_morgan_left(p, q):
        return NOT(AND(p)(q))
    
    def de_morgan_right(p, q):
        return OR(NOT(p))(NOT(q))
    
    print(show_bool(BEQ(de_morgan_left(TRUE, FALSE))(de_morgan_right(TRUE, FALSE))))  # TRUE
    print(show_bool(BEQ(de_morgan_left(TRUE, TRUE))(de_morgan_right(TRUE, TRUE))))    # TRUE
    print(show_bool(BEQ(de_morgan_left(FALSE, TRUE))(de_morgan_right(FALSE, TRUE))))  # TRUE
    print(show_bool(BEQ(de_morgan_left(FALSE, FALSE))(de_morgan_right(FALSE, FALSE))))# TRUE
    
    print("\n=== CHURCH NUMERALS ===")
    print(f"{jsnum(ZERO)} {jsnum(ONE)} {jsnum(TWO)} {jsnum(THREE)}")  # 0 1 2 3
    print(show_bool(once(NOT)(TRUE)))      # FALSE
    print(show_bool(twice(NOT)(FALSE)))    # FALSE
    print(show_bool(thrice(NOT)(TRUE)))    # FALSE
    print(show_bool(fourfold(NOT)(FALSE))) # FALSE
    print(show_bool(fivefold(NOT)(TRUE)))  # FALSE
    
    print("\n=== ARITHMETIC ===")
    print(jsnum(PLUS(TWO)(THREE)))         # 5
    print(jsnum(MULT(TWO)(THREE)))         # 6
    print(jsnum(POW(TWO)(THREE)))          # 8
    print(jsnum(PRED(THREE)))              # 2
    print(jsnum(SUB(FIVE)(TWO)))           # 3
    
    print("\n=== PREDICATES ===")
    print(show_bool(ISZERO(ZERO)))         # TRUE
    print(show_bool(ISZERO(ONE)))          # FALSE
    print(show_bool(LEQ(TWO)(THREE)))      # TRUE
    print(show_bool(EQN(PLUS(TWO)(THREE))(FIVE)))  # TRUE
    
    print("\n=== EXPRESSION TRANSLATION ===")
    # !x == y || (a && z)
    print(show_bool(EXPR(TRUE, FALSE, TRUE, TRUE)))   # TRUE
    print(show_bool(EXPR(FALSE, TRUE, FALSE, TRUE)))  # TRUE
    print(show_bool(EXPR(TRUE, TRUE, TRUE, FALSE)))   # FALSE

"""
REFERENCE MAPPING (Python):

| Math                      | Name     | Python                                        |
|---------------------------|----------|-----------------------------------------------|
| λa.a                      | I        | I = lambda a: a                               |
| λa.λb.a                   | K        | K = lambda a: lambda b: a                     |
| λa.λb.b                   | KI       | KI = lambda a: lambda b: b                    |
| λf.λa.λb.f b a            | C        | C = lambda f: lambda a: lambda b: f(b)(a)     |
| λf.λg.λx.f x (g x)        | S        | S = lambda f: lambda g: lambda x: f(x)(g(x))  |
| λf.λg.λx.f (g x)          | B        | B = lambda f: lambda g: lambda x: f(g(x))     |
| λf.λx.f x x               | W        | W = lambda f: lambda x: f(x)(x)               |
| λa.λf.f a                 | T        | T = lambda a: lambda f: f(a)                  |
| λf.f f                    | M        | M = lambda f: f(f)                            |
| λp.p FALSE TRUE           | NOT      | NOT = lambda p: p(FALSE)(TRUE)                |
| λp.λq.p q FALSE           | AND      | AND = lambda p: lambda q: p(q)(FALSE)         |
| λp.λq.p TRUE q            | OR       | OR = lambda p: lambda q: p(TRUE)(q)           |
| λf.λx.x                   | ZERO     | ZERO = lambda f: lambda x: x                  |
| λn.λf.λx.f (n f x)        | SUCC     | SUCC = lambda n: lambda f: lambda x: f(n(f)(x)) |
| λm.λn.λf.λx.m f (n f x)   | PLUS     | PLUS = lambda m: lambda n: lambda f: lambda x: m(f)(n(f)(x)) |
| λm.λn.λf.m (n f)          | MULT     | MULT = lambda m: lambda n: lambda f: m(n(f)) |
| λm.λn.n m                 | POW      | POW = lambda m: lambda n: n(m)                |
"""