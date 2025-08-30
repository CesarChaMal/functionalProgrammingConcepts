/*
λ-Calculus + Combinators + Church Arithmetic (TypeScript)

A comprehensive, runnable reference for λ-calculus fundamentals including:
* Core combinators (I, K, KI, C, S, B, W, T, M) with .inspect()
* Church booleans and logic (NOT, AND, OR, equality)
* DeMorgan verification
* Church numerals (0, 1, 2, 3, …) + iteration helpers
* Arithmetic: successor, addition, multiplication, exponentiation, predecessor
* Pairs (for predecessor implementation)
* Predicates: isZero, leq, eq
* Translation of !x == y || (a && z) into λ-calculus
*/

// YouTube videos

// Lambda Calculus - Fundamentals of Lambda Calculus & Functional Programming in JavaScript
// [https://www.youtube.com/watch?v=3VQ382QG-y4](https://www.youtube.com/watch?v=3VQ382QG-y4)

// A Flock of Functions; Combinators, Lambda Calculus, & Church Encodings in JS - Part II
// [https://www.youtube.com/watch?v=pAnLQ9jwN-E](https://www.youtube.com/watch?v=pAnLQ9jwN-E)

// Online IDE
// [https://www.programiz.com/javascript/online-compiler/](https://www.programiz.com/javascript/online-compiler/)

// === HELPERS ===

// Convert Church numeral to JS number
const jsnum = (n: any) => n((x: number) => x + 1)(0)

// Convert Church boolean to string for display
const showBool = (p: any) => p('TRUE')('FALSE')

// Attach inspector to Church numerals
const withInspect = (n: any, label: string) => {
    n.inspect = () => `${label} = ${jsnum(n)}`
    return n
}

// === 1) CORE COMBINATORS ===

// Identity (I) — λa.a
const I = (a: any) => a
I.inspect = () => 'I (identity)'

// Kestrel (K) — λa.λb.a (TRUE)
const K = (a: any) => (b: any) => a
K.inspect = () => 'K (kestrel / true)'

// Kite (KI) — λa.λb.b (FALSE)
const KI = (a: any) => (b: any) => b
KI.inspect = () => 'KI (kite / false)'

// Cardinal (C) — λf.λa.λb.f b a (flip)
const C = (f: any) => (a: any) => (b: any) => f(b)(a)
C.inspect = () => 'C (cardinal / flip)'

// Starling (S) — λf.λg.λx.f x (g x)
const S = (f: any) => (g: any) => (x: any) => f(x)(g(x))
S.inspect = () => 'S (starling)'

// Bluebird (B) — λf.λg.λx.f (g x) (compose)
const B = (f: any) => (g: any) => (x: any) => f(g(x))
B.inspect = () => 'B (bluebird / compose)'

// Warbler (W) — λf.λx.f x x
const W = (f: any) => (x: any) => f(x)(x)
W.inspect = () => 'W (warbler)'

// Thrush (T) — λa.λf.f a
const T = (a: any) => (f: any) => f(a)
T.inspect = () => 'T (thrush)'

// Mockingbird (M) — λf.f f (self-application)
const M = (f: any) => f(f)
M.inspect = () => 'M (mockingbird)'

// === 2) CHURCH BOOLEANS ===

// TRUE and FALSE
const TRUE = K
TRUE.inspect = () => 'TRUE / K'
const FALSE = KI
FALSE.inspect = () => 'FALSE / KI'

// NOT — λp.p FALSE TRUE
const NOT = (p: any) => p(FALSE)(TRUE)
NOT.inspect = () => 'NOT'

// AND — λp.λq.p q FALSE
const AND = (p: any) => (q: any) => p(q)(FALSE)
AND.inspect = () => 'AND (p q FALSE)'

// AND (alternative) — λp.λq.p q p
const AND2 = (p: any) => (q: any) => p(q)(p)
AND2.inspect = () => 'AND (alt: p q p)'

// OR — λp.λq.p TRUE q
const OR = (p: any) => (q: any) => p(TRUE)(q)
OR.inspect = () => 'OR (p TRUE q)'

// OR (alternative) — λp.λq.p p q
const OR2 = (p: any) => (q: any) => p(p)(q)
OR2.inspect = () => 'OR (alt: p p q)'

// Boolean equality — λp.λq.p q (NOT q)
const BEQ = (p: any) => (q: any) => p(q)(NOT(q))
BEQ.inspect = () => 'BEQ (boolean equality)'

// === 3) CHURCH NUMERALS ===

// Zero — λf.λx.x
const ZERO = (f: any) => (x: any) => x
ZERO.inspect = () => '0'

// Successor — λn.λf.λx.f (n f x)
const SUCC = (n: any) => (f: any) => (x: any) => f(n(f)(x))
SUCC.inspect = () => 'SUCC'

// Build numerals
const ONE = withInspect(SUCC(ZERO), '1')
const TWO = withInspect(SUCC(ONE), '2')
const THREE = withInspect(SUCC(TWO), '3')
const FOUR = withInspect(SUCC(THREE), '4')
const FIVE = withInspect(SUCC(FOUR), '5')

// Iteration helpers
const once = (f: any) => (a: any) => f(a)
const twice = (f: any) => (a: any) => f(f(a))
const thrice = (f: any) => (a: any) => f(f(f(a)))
const fourfold = (f: any) => (a: any) => f(f(f(f(a))))
const fivefold = (f: any) => (a: any) => f(f(f(f(f(a)))))

// === 4) PAIRS (for predecessor) ===

// Church pair — λa.λb.λf.f a b
const PAIR = (a: any) => (b: any) => (f: any) => f(a)(b)
const FST = (p: any) => p(K)
const SND = (p: any) => p(KI)

// === 5) ARITHMETIC ===

// Addition — λm.λn.λf.λx.m f (n f x)
const PLUS = (m: any) => (n: any) => (f: any) => (x: any) => m(f)(n(f)(x))
PLUS.inspect = () => 'PLUS'

// Multiplication — λm.λn.λf.m (n f)
const MULT = (m: any) => (n: any) => (f: any) => m(n(f))
MULT.inspect = () => 'MULT'

// Exponentiation — λm.λn.n m
const POW = (m: any) => (n: any) => n(m)
POW.inspect = () => 'POW'

// Predecessor (via pairs)
const STEP = (p: any) => PAIR(SND(p))(SUCC(SND(p)))
const PRED = (n: any) => FST(n(STEP)(PAIR(ZERO)(ZERO)))
PRED.inspect = () => 'PRED'

// Subtraction — λm.λn.n PRED m
const SUB = (m: any) => (n: any) => n(PRED)(m)
SUB.inspect = () => 'SUB'

// === 6) PREDICATES ===

// Is zero — λn.n (λ_.FALSE) TRUE
const ISZERO = (n: any) => n((_: any) => FALSE)(TRUE)
ISZERO.inspect = () => 'ISZERO'

// Less than or equal — λm.λn.ISZERO (SUB m n)
const LEQ = (m: any) => (n: any) => ISZERO(SUB(m)(n))
LEQ.inspect = () => 'LEQ'

// Numeric equality — λm.λn.AND (LEQ m n) (LEQ n m)
const EQN = (m: any) => (n: any) => AND(LEQ(m)(n))(LEQ(n)(m))
EQN.inspect = () => 'EQN'

// === 7) EXPRESSION TRANSLATION ===

// !x == y || (a && z) becomes:
const EXPR = (x: any, y: any, a: any, z: any) => OR(BEQ(NOT(x))(y))(AND(a)(z))

// === DEMONSTRATIONS ===

console.log('=== COMBINATORS ===')
console.log(I(42))                    // 42
console.log(K(1)(2))                  // 1
console.log(C(K)(1)(2))              // 2
console.log(M(I))                     // I

console.log('\n=== BOOLEANS ===')
console.log(showBool(TRUE))           // TRUE
console.log(showBool(FALSE))          // FALSE
console.log(showBool(NOT(TRUE)))      // FALSE
console.log(showBool(AND(TRUE)(FALSE))) // FALSE
console.log(showBool(OR(FALSE)(TRUE)))  // TRUE

console.log('\n=== DEMORGAN VERIFICATION ===')
// ¬(p ∧ q) ≡ (¬p) ∨ (¬q)
const deMorganLeft = (p: any, q: any) => NOT(AND(p)(q))
const deMorganRight = (p: any, q: any) => OR(NOT(p))(NOT(q))
console.log(showBool(BEQ(deMorganLeft(TRUE, FALSE))(deMorganRight(TRUE, FALSE))))  // TRUE
console.log(showBool(BEQ(deMorganLeft(TRUE, TRUE))(deMorganRight(TRUE, TRUE))))    // TRUE

console.log('\n=== CHURCH NUMERALS ===')
console.log(jsnum(ZERO), jsnum(ONE), jsnum(TWO), jsnum(THREE)) // 0 1 2 3
console.log(showBool(once(NOT)(TRUE)))      // FALSE
console.log(showBool(twice(NOT)(FALSE)))    // FALSE

console.log('\n=== ARITHMETIC ===')
console.log(jsnum(PLUS(TWO)(THREE)))       // 5
console.log(jsnum(MULT(TWO)(THREE)))       // 6
console.log(jsnum(POW(TWO)(THREE)))        // 8
console.log(jsnum(PRED(THREE)))            // 2
console.log(jsnum(SUB(FIVE)(TWO)))         // 3

console.log('\n=== PREDICATES ===')
console.log(showBool(ISZERO(ZERO)))        // TRUE
console.log(showBool(ISZERO(ONE)))         // FALSE
console.log(showBool(LEQ(TWO)(THREE)))     // TRUE
console.log(showBool(EQN(PLUS(TWO)(THREE))(FIVE))) // TRUE

console.log('\n=== EXPRESSION TRANSLATION ===')
// !x == y || (a && z)
console.log(showBool(EXPR(TRUE, FALSE, TRUE, TRUE)))   // TRUE
console.log(showBool(EXPR(FALSE, TRUE, FALSE, TRUE)))  // TRUE
console.log(showBool(EXPR(TRUE, TRUE, TRUE, FALSE)))   // FALSE

/*
REFERENCE MAPPING:

| Math                      | Name     | TypeScript                                    |
|---------------------------|----------|-----------------------------------------------|
| λa.a                      | I        | const I = a => a                              |
| λa.λb.a                   | K        | const K = a => b => a                         |
| λa.λb.b                   | KI       | const KI = a => b => b                        |
| λf.λa.λb.f b a            | C        | const C = f => a => b => f(b)(a)              |
| λf.λg.λx.f x (g x)        | S        | const S = f => g => x => f(x)(g(x))           |
| λf.λg.λx.f (g x)          | B        | const B = f => g => x => f(g(x))              |
| λf.λx.f x x               | W        | const W = f => x => f(x)(x)                   |
| λa.λf.f a                 | T        | const T = a => f => f(a)                      |
| λf.f f                    | M        | const M = f => f(f)                           |
| λp.p FALSE TRUE           | NOT      | const NOT = p => p(FALSE)(TRUE)               |
| λp.λq.p q FALSE           | AND      | const AND = p => q => p(q)(FALSE)             |
| λp.λq.p TRUE q            | OR       | const OR = p => q => p(TRUE)(q)               |
| λf.λx.x                   | ZERO     | const ZERO = f => x => x                      |
| λn.λf.λx.f (n f x)        | SUCC     | const SUCC = n => f => x => f(n(f)(x))        |
| λm.λn.λf.λx.m f (n f x)   | PLUS     | const PLUS = m => n => f => x => m(f)(n(f)(x))|
| λm.λn.λf.m (n f)          | MULT     | const MULT = m => n => f => m(n(f))           |
| λm.λn.n m                 | POW      | const POW = m => n => n(m)                    |
*/