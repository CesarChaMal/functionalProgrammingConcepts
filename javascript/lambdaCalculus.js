/*
λ-Calculus + Combinators + Church Arithmetic (JavaScript)

A comprehensive, runnable reference for λ-calculus fundamentals including:
* Core combinators (I, K, KI, C, S, B, W, T, M)
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
const jsnum = n => n(x => x + 1)(0)

// Convert Church boolean to string
const showBool = p => p('TRUE')('FALSE')

// === 1) CORE COMBINATORS ===

// Identity (I) — λa.a
const I = a => a

// Kestrel (K) — λa.λb.a (TRUE)
const K = a => b => a

// Kite (KI) — λa.λb.b (FALSE)
const KI = a => b => b

// Cardinal (C) — λf.λa.λb.f b a (flip)
const C = f => a => b => f(b)(a)

// Starling (S) — λf.λg.λx.f x (g x)
const S = f => g => x => f(x)(g(x))

// Bluebird (B) — λf.λg.λx.f (g x) (compose)
const B = f => g => x => f(g(x))

// Warbler (W) — λf.λx.f x x
const W = f => x => f(x)(x)

// Thrush (T) — λa.λf.f a
const T = a => f => f(a)

// Mockingbird (M) — λf.f f (self-application)
const M = f => f(f)

// === 2) CHURCH BOOLEANS ===

// TRUE and FALSE
const TRUE = K
const FALSE = KI

// NOT — λp.p FALSE TRUE
const NOT = p => p(FALSE)(TRUE)

// AND — λp.λq.p q FALSE
const AND = p => q => p(q)(FALSE)

// OR — λp.λq.p TRUE q
const OR = p => q => p(TRUE)(q)

// Boolean equality — λp.λq.p q (NOT q)
const BEQ = p => q => p(q)(NOT(q))

// === 3) CHURCH NUMERALS ===

// Zero — λf.λx.x
const ZERO = f => x => x

// Successor — λn.λf.λx.f (n f x)
const SUCC = n => f => x => f(n(f)(x))

// Build numerals
const ONE = SUCC(ZERO)
const TWO = SUCC(ONE)
const THREE = SUCC(TWO)
const FOUR = SUCC(THREE)
const FIVE = SUCC(FOUR)

// === 4) PAIRS (for predecessor) ===

// Church pair — λa.λb.λf.f a b
const PAIR = a => b => f => f(a)(b)
const FST = p => p(K)
const SND = p => p(KI)

// === 5) ARITHMETIC ===

// Addition — λm.λn.λf.λx.m f (n f x)
const PLUS = m => n => f => x => m(f)(n(f)(x))

// Multiplication — λm.λn.λf.m (n f)
const MULT = m => n => f => m(n(f))

// Exponentiation — λm.λn.n m
const POW = m => n => n(m)

// Predecessor (via pairs)
const STEP = p => PAIR(SND(p))(SUCC(SND(p)))
const PRED = n => FST(n(STEP)(PAIR(ZERO)(ZERO)))

// Subtraction — λm.λn.n PRED m
const SUB = m => n => n(PRED)(m)

// === 6) PREDICATES ===

// Is zero — λn.n (λ_.FALSE) TRUE
const ISZERO = n => n(_ => FALSE)(TRUE)

// Less than or equal — λm.λn.ISZERO (SUB m n)
const LEQ = m => n => ISZERO(SUB(m)(n))

// Numeric equality — λm.λn.AND (LEQ m n) (LEQ n m)
const EQN = m => n => AND(LEQ(m)(n))(LEQ(n)(m))

// === 7) EXPRESSION TRANSLATION ===

// !x == y || (a && z) becomes:
const EXPR = (x, y, a, z) => OR(BEQ(NOT(x))(y))(AND(a)(z))

// === DEMONSTRATIONS ===

console.log('=== COMBINATORS ===')
console.log(I(42))                    // 42
console.log(K(1)(2))                  // 1
console.log(C(K)(1)(2))              // 2

console.log('\n=== BOOLEANS ===')
console.log(showBool(TRUE))           // TRUE
console.log(showBool(FALSE))          // FALSE
console.log(showBool(NOT(TRUE)))      // FALSE
console.log(showBool(AND(TRUE)(FALSE))) // FALSE
console.log(showBool(OR(FALSE)(TRUE)))  // TRUE

console.log('\n=== DEMORGAN VERIFICATION ===')
// ¬(p ∧ q) ≡ (¬p) ∨ (¬q)
const deMorganLeft = (p, q) => NOT(AND(p)(q))
const deMorganRight = (p, q) => OR(NOT(p))(NOT(q))
console.log(showBool(BEQ(deMorganLeft(TRUE, FALSE))(deMorganRight(TRUE, FALSE))))  // TRUE
console.log(showBool(BEQ(deMorganLeft(TRUE, TRUE))(deMorganRight(TRUE, TRUE))))    // TRUE

console.log('\n=== CHURCH NUMERALS ===')
console.log(jsnum(ZERO), jsnum(ONE), jsnum(TWO), jsnum(THREE)) // 0 1 2 3

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
REFERENCE MAPPING (JavaScript):

| Math                      | Name     | JavaScript                                    |
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
| λn.n (λ_.FALSE) TRUE      | ISZERO   | const ISZERO = n => n(_ => FALSE)(TRUE)       |
*/