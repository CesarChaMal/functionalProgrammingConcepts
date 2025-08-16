/*
 FP CONCEPTS (TypeScript) — ALL IN ONE, ANNOTATED
 1) Lambda, Application, Currying, Partial Application — lambdas and closures.
 2) Composition (∘) — compose(f, g)(x) == f(g(x)).
 3) Referential Transparency — pure expressions can be substituted by values.
 4) Immutability — prefer new objects (spreads) vs mutation.
 5) Higher‑Order Functions — array map/filter/reduce pipelines.
 6) Functor — Array.map, Option.map (custom).
 7) Applicative — ap / liftA2 for Option (independent combination).
 8) Monad — flatMap for Option; Promise.then is also a Monad.
 9) Natural Transformation — Array<A> -> Option<A> (head).
 10) Monoid — reduce with associative op and identity.
 11) ADTs & Pattern Matching — discriminated unions + exhaustive handling.
 12) Effects at the Edges — keep core pure; do I/O at boundary.
 13) Property‑Based Testing — outline with fast-check.
*/

// 1) Lambda, Application, Currying, Partial Application
const inc = (x: number) => x + 1;
const add = (x: number) => (y: number) => x + y; // curried
const add5 = add(5);                              // partial
const seven = add5(2);                            // 7
/*
1. Line 1: a simple lambda (a function value)

- Code: const inc = (x: number) => x + 1
    - Meaning:
- inc is a function from number to number.
- It’s an arrow function (lambda) that returns x + 1.

- Usage:
- inc(10) === 11

- Notes:
- Pure function: same input → same output, no side effects.

2. Line 2: a curried function (a function that returns a function)

- Code: const add = (x: number) => (y: number) => x + y
    - Meaning:
- add takes one argument x and returns a new function that takes y and computes x + y.
- This is currying: (x, y) => x + y becomes x => (y => x + y).

    - Why this is useful:
    - Enables partial application (fix some args now, pass the rest later).
- Composes nicely with other higher‑order functions.

- Closure:
- The inner function (y => x + y) “remembers” x via closure.

3. Line 3: partial application (fix the first argument)

- Code: const add5 = add(5)
    - Meaning:
- Provide x = 5 to add, producing a new function y => 5 + y.
- add5 is now a function number => number.

- Usage:
- add5(10) === 15

4. Line 4: function application (call the specialized function)

- Code: const seven = add5(2)
    - Meaning:
- Calls add5 with y = 2 → computes 5 + 2 = 7.
    - seven === 7

Equivalences and tips
- Direct two‑step application:
    - add(5)(2) === 7

    - Uncurried vs curried:
    - Uncurried version would be const add2 = (x: number, y: number) => x + y
    - Currying lets you pre-fill arguments and treat the result as a first‑class function.

- Types:
- You can annotate explicitly if you like:
    - const add: (x: number) => (y: number) => number = x => y => x + y
*/

// 2) Composition (∘)
const compose = <A,B,C>(f: (b:B)=>C, g:(a:A)=>B) => (a:A) => f(g(a));
const double = (x: number) => x * 2;
const compRes = compose(inc, double)(10);         // 21
/*
1. Line 1: define a generic compose

- Code: const compose = <A,B,C>(f, g) => a => f(g(a))
    - Meaning:
- Given g: A -> B and f: B -> C, compose returns a new function h: A -> C such that h(a) = f(g(a)).
    - It builds a pipeline where g runs first, then f.

2. Line 2: define a “double” function

- Code: const double = (x: number) => x * 2
    - Meaning:
- double maps a number to its double.

- Usage:
- double(7) === 14

3. Line 3: apply the composed function

- Code: const compRes = compose(inc, double)(10)
    - Step-by-step trace:
    1. compose(inc, double) creates h(a) = inc(double(a)).
2. h(10) → double(10) = 20
3. inc(20) = 21
4. compRes === 21

- Assumption:
- inc is defined (e.g., const inc = (x: number) => x + 1).

Extra notes
- Order matters:
    - compose(f, g)(x) = f(g(x)) means “apply g first, then f”.

- Types in this instance:
    - double: (number) => number
    - inc: (number) => number
    - compose(inc, double): (number) => number

    - Mental model:
    - Composition wires small functions together so the output of one feeds the next, without intermediate variables.
*/

// 3) Referential Transparency
const pureTotal = [1,2,3].reduce((a,b)=>a+b, 0);  // == 6
// console.log("hi"); // I/O → not referentially transparent
/*
1. What “referential transparency” means

- An expression is referentially transparent if you can replace it with its value everywhere in the program without changing the program’s behavior.
- In practice: no side effects, no hidden/external state, deterministic (same input → same output).

2. The pure expression

- Code: [1,2,3].reduce((a,b)=>a+b, 0)
- Meaning:
- Builds an array [1, 2, 3].
- reduce folds the elements with the function (a, b) => a + b starting from 0.

- Micro-trace:
- Start: acc = 0
    - Step 1: 0 + 1 = 1
    - Step 2: 1 + 2 = 3
    - Step 3: 3 + 3 = 6
    - Result: 6

- Why it’s referentially transparent:
    - No I/O, no mutation, no randomness.
- You can safely replace [1,2,3].reduce((a,b)=>a+b, 0) with 6 anywhere; behavior won’t change.

- Therefore: pureTotal is 6; writing const pureTotal = 6 is behaviorally equivalent.

3. The side‑effecting expression (not referentially transparent)

- Code: console.log("hi")
- Why it’s not referentially transparent:
    - It performs I/O by printing to stdout.
- Replacing it with its “value” (undefined) would remove the print, changing observable behavior.
- So it cannot be freely substituted by a value without altering the program.

    Key takeaways
- Pure expressions are interchangeable with their values—great for reasoning, testing, and refactoring.
- Side effects (like printing) are not referentially transparent; they affect the outside world and cannot be replaced by a value without changing behavior.
*/

// 4) Immutability
type Point = Readonly<{ x: number; y: number }>;
const p1: Point = { x: 1, y: 1 };
const p2: Point = { ...p1, x: 2 };                // new object; p1 unchanged
/*
1. Line 1: define an immutable data carrier (a Readonly object type)

- Code: type Point = Readonly<{ x: number; y: number }>;
- What Readonly does:
    - Creates a type where all properties are read‑only at the type level.
- Prevents assignments like p.x = ... in TypeScript code (compile‑time error).
- Works with structural typing; any object with x and y numbers can be treated as Point if it satisfies the readonly constraint.

- Immutability here:
    - The compiler forbids property reassignment through this Point type.
- Note: this is shallow and enforced by the type system only; at runtime the object is still a plain JS object (not frozen). Using numbers here is fully safe for value semantics.

2. Line 2: create the first instance

- Code: const p1: Point = { x: 1, y: 1 };
- Meaning:
- Constructs a value with fields x = 1, y = 1 typed as Point.
- Attempting p1.x = 42 would be a compile‑time error (read‑only property).

3. Line 3: create a new instance instead of mutating

- Code: const p2: Point = { ...p1, x: 2 };
- Step‑by‑step:
    1. Spread p1 to copy its properties into a new object literal (shallow copy).
2. Override x to 2 while leaving y as p1.y (1).
3. The result is a new object { x: 2, y: 1 } assigned to p2.

- Key idea:
    - You didn’t “change” p1; you built a new value p2 with a different x.
- p1 remains { x: 1, y: 1 }. p2 is { x: 2, y: 1 }.

Why this matters in FP
- Immutability avoids accidental shared‑state bugs.
- Values are safe to pass around and reuse; reasoning is simpler (no hidden changes).
- Plays well with concurrency and caching.
- When you need a “modified” version, build a new value via spreads (as shown with p2), rather than mutating in place.
*/

// 5) Higher‑Order Functions
const hofSum = [1, 2, 3]
    .map(x => x * 2)
    .filter(x => x > 2)
    .reduce((a, b) => a + b, 0); // 10
/*
1. Built-in HOF pipeline (map → filter → reduce)
- What happens:
  - Start with [1, 2, 3]
  - map(x => x*2)       → [2, 4, 6]
  - filter(x => x > 2)  → [4, 6]
  - reduce((a,b)=>a+b) with 0 → 0 + 4 + 6 = 10

- Why these are HOFs:
  - map, filter, reduce each take functions (lambdas) as arguments, so they’re higher‑order functions.
- Complexity:
  - Each stage is linear; in practice you traverse the array multiple times (once per stage).
- Purity note:
  - If the lambdas are pure, the pipeline is pure and deterministic.
*/

// Custom HOF #1: map for Array (re-implement to show the idea)
function mapArray<A, B>(xs: A[], f: (a: A) => B): B[] {
    const out: B[] = new Array(xs.length);
    for (let i = 0; i < xs.length; i++) out[i] = f(xs[i]);
    return out;
}
// Usage: doubles each element -> [2, 4, 6]
const customDoubled: number[] = mapArray([1, 2, 3], x => x * 2);
/*
2. Custom HOF #1: mapArray

- Purpose:
  - Transform each element independently using a provided function, producing a new array of possibly different element type.

- Signature reasoning:
  - mapArray<A, B>(xs: A[], f: (a: A) => B): B[]
  - A is input element type; B is output element type; f captures the behavior.

- Behavior (step-by-step):
  - Allocate out with the same length as xs.
  - For i in [0..xs.length):
    - out[i] = f(xs[i])
  - Return out.

- Example:
  - mapArray([1,2,3], x => x * 2) → [2, 4, 6]
  - mapArray(["a","bb"], s => s.length) → [1, 2]

- Complexity:
  - Time O(n), Space O(n) for the new array.

- Purity and referential transparency:
  - If f is pure, mapArray is pure; results depend only on xs and f.

- Why it’s higher‑order:
  - It takes a function f as an argument.

- Variations:
  - Implement lazily via generators/iterables to defer work.
  - Parallel mapping if f is pure and independent per element.

- Edge cases:
  - Empty input → returns [].
  - Avoid side effects in f for better reasoning and testability.
*/

// Custom HOF #2: map -> filter -> reduce as one reusable helper
function mapFilterReduce<A, B>(
    xs: A[],
    mapper: (a: A) => A,
    predicate: (a: A) => boolean,
    zero: B,
    reducer: (b: B, a: A) => B
): B {
    let acc: B = zero;
    for (const a of xs) {
        const m = mapper(a);
        if (predicate(m)) acc = reducer(acc, m);
    }
    return acc;
}
// Usage: same logic as the pipeline above -> 10
const customMfr: number = mapFilterReduce(
    [1, 2, 3],
    x => x * 2,      // [2, 4, 6]
    x => x > 2,      // [4, 6]
    0,
    (a, b) => a + b  // 10
);
/*
3. Custom HOF #2: mapFilterReduce (one reusable helper)

- Purpose:
  - Perform a “map → filter → reduce” pipeline in a single pass, avoiding intermediate arrays.

- Signature reasoning:
  - mapper: (A) => A keeps one element type flowing through; you can generalize to (A) => C by adjusting predicate/reducer generics.
  - zero: initial accumulator value of type B.
  - reducer: (B, A) => B folds kept elements into the accumulator.

- Behavior (step-by-step):
  - acc = zero
  - For each a in xs:
    - m = mapper(a)           // map step
    - if predicate(m):        // filter step
        acc = reducer(acc, m) // reduce step
  - Return acc

- Example:
  - xs = [1,2,3]
  - mapper = x*2 → [2,4,6]
  - predicate = x>2 → keep [4,6]
  - zero = 0
  - reducer = (a,b)=>a+b → 10

- Complexity:
  - Single traversal: Time O(n), Space O(1) extra.

- Purity and referential transparency:
  - If mapper, predicate, reducer are pure, this helper is pure and deterministic.

- Why it’s higher‑order:
  - Accepts three behaviors (mapper, predicate, reducer).

- When to use:
  - You need the clarity of a pipeline with the efficiency of one pass.

- Variations:
  - Generalize mapper to change the element type.
  - Add short‑circuiting by encoding early stop in the accumulator (e.g., using a tagged result).

- Edge cases:
  - [] → zero.
  - predicate always false → zero.
  - Non‑associative reducers are allowed, but prefer associative reducers for predictable behavior.
*/

// Custom HOF #3: return a new function by repeating an operation n times
function repeat<T>(n: number, f: (t: T) => T): (t: T) => T {
    return (t: T) => {
        let r = t;
        for (let i = 0; i < n; i++) r = f(r);
        return r;
    };
}
// Usage: reuse an `inc` function, apply it 3 times
// const inc = (x: number) => x + 1;
const inc3 = repeat(3, inc);
const inc3Res = inc3(10); // 13
/*
4. Custom HOF #3: repeat (returns a new function)

- Purpose:
  - Build a new function that applies f to its input n times:
    repeat(n, f)(x) = f(f(...f(x)...)) with n applications.

- Signature reasoning:
  - f must be an endofunction on T (T => T) so it composes with itself.

- Behavior (step-by-step):
  - Return a function that:
    - Starts r = t
    - Loops i in [0..n): r = f(r)
    - Returns r

- Examples:
  - inc = x => x + 1; repeat(3, inc)(10) → 13
  - exclaim = (s: string) => s + "!"; repeat(3, exclaim)("go") → "go!!!"

- Complexity:
  - Each call to the returned function applies f n times (O(n)).

- Purity and referential transparency:
  - If f is pure, repeat returns a pure function; same input → same output.

- Why it’s higher‑order:
  - Takes a function and returns a new function.

- Edge cases:
  - n = 0 → identity function (returns input unchanged).
  - Large n with expensive f → consider precomposing once or algebraic shortcuts if available.
*/

// Option helpers (for 6–9)
type Option<T> = { _tag: 'Some'; value: T } | { _tag: 'None' };
const Some = <T>(value: T): Option<T> => ({ _tag: 'Some', value });
const None: Option<never> = { _tag: 'None' };
const isSome = <T>(o: Option<T>): o is { _tag: 'Some'; value: T } => o._tag === 'Some';

// 6) Functor: map preserves structure
const map = <A,B>(oa: Option<A>, f: (a: A) => B): Option<B> => isSome(oa) ? Some(f(oa.value)) : None;
const fArr = [1,2,3].map(x => x+1);               // [2,3,4]
const fOpt = map(Some(42), x => x + 1);           // Some(43)
/*
-------------------------------------------------------------------------------
Explanation of Functor behavior in TypeScript map
-------------------------------------------------------------------------------
1. Code lines:
   - const fArr = [1,2,3].map(x => x+1);
   - const fOpt = map(Some(42), x => x + 1);

   Meaning:
   - For arrays: map applies the function to each element, producing a new array.
   - For Option: map applies the function to the contained value if present, otherwise keeps None.

2. Functor definition (informal):
   A type constructor F<_> that supports a structure-preserving mapping:
     map: (A -> B) -> F<A> -> F<B>
   - For Array:
     F = Array
     map(f, Array<A>) -> Array<B>
   - For Option:
     F = Option
     map(f, Option<A>) -> Option<B>
   - "Preserve structure" means: array length is same, Option remains Some/None.

3. Why useful:
   - For arrays: concise transformations without explicit loops.
   - For Option: eliminates manual null/undefined checks. Instead of:
       if (isSome(opt)) return Some(f(opt.value)); else return None;
     you simply write:
       map(opt, f)

4. Behavior summary:
   - Array:
       [1,2,3].map(x => x+1) = [2,3,4]
   - Option:
       map(Some(42), x => x+1) = Some(43)
       map(None, x => x+1)     = None

5. Functor laws:
   - Identity:       map(fa, x => x) == fa
   - Composition:    map(fa, x => g(f(x))) == map(map(fa,f), g)
     (for pure functions f and g)

6. Purity:
   - If f is pure, map is pure and referentially transparent.
   - Same input + same function => same output.

7. Edge cases:
   - Array: works on empty array → returns empty array.
   - Option: None mapped → stays None.
   - Performance: O(n) for array of n elements; O(1) for Option.

8. Chaining examples:
   - Arrays:
       [10].map(x=>x+1).map(x=>x*2) = [22]
   - Option:
       map(Some(10), x=>x+1) = Some(11)
       map(Some(11), x=>x*2) = Some(22)
       map(None, x=>x+1)     = None
-------------------------------------------------------------------------------
*/

// 7) Applicative: ap / liftA2 combine independent contexts
const ap = <A,B>(of: Option<(a:A)=>B>, oa: Option<A>): Option<B> => isSome(of) && isSome(oa) ? Some(of.value(oa.value)) : None;
const pure = <T>(x: T): Option<T> => Some(x);
const liftA2 = <A,B,C>(f: (a:A, b:B)=>C) => (oa: Option<A>) => (ob: Option<B>) => ap(ap(pure((a:A)=> (b:B)=> f(a,b)), oa), ob);
const name: Option<string> = Some('Ada');
const age:  Option<number> = Some(36);
const user = liftA2((n:string, a:number) => ({ n, a }))(name)(age); // Some({n:'Ada',a:36})
/*
1. Code lines:
- ap: Apply an Option-wrapped function to an Option-wrapped value.
  - If either is None → result is None.
  - If both present → call the function with the value.

- pure: Lift a raw value into the Option context (wraps in Some).

- liftA2: Lifts a binary function (A,B)=>C into the Option context.
  - Curries the function into (A)=>(B)=>C.
  - Uses ap twice: first to apply the function to oa, then to ob.

2. Why Applicative?
- Applicative combines independent Option values without sequencing.
- "Independent" means: neither Option depends on the other being computed first.
- Signature: liftA2 :: (A,B)=>C -> Option<A> -> Option<B> -> Option<C>.

3. Step-by-step evaluation of user:
- pure((a)=> (b)=> f(a,b)) lifts a curried function into Option.
- ap(..., name) applies it to Some("Ada") → Some((b)=> f("Ada", b)).
- ap(..., age) applies that to Some(36) → Some(f("Ada",36)).
- f("Ada",36) = {n:"Ada", a:36}.
- Final result: Some({n:"Ada",a:36}).

4. Behavior summary:
- If name or age is None → result is None.
- If both present → function runs and wraps result in Some.

5. Use cases:
- Building objects from multiple Option values.
- Safe combination of independent inputs without null-checks.

6. Purity:
- If f is pure → liftA2 is deterministic and referentially transparent.

7. Edge cases:
- None propagates: short-circuit → no function call.
- Functions must be curried for ap to apply them one argument at a time.

8. Laws (Applicative intuition):
- Identity: ap(pure(x=>x), v) == v.
- Homomorphism: ap(pure(f), pure(x)) == pure(f(x)).
- Interchange: ap(u, pure(y)) == ap(pure(f=>f(y)), u).
*/

// 8) Monad: flatMap for dependent sequencing; Promise.then also monadic
const flatMap = <A,B>(oa: Option<A>, f: (a:A) => Option<B>): Option<B> => isSome(oa) ? f(oa.value) : None;
const monadRes = flatMap(Some(2), x => map(Some(3), y => x + y));     // Some(5)
// Promise.resolve(2).then(x => Promise.resolve(3).then(y => x+y));
/*
1. Line: flatMap sequences dependent computations (Monad behavior)

- Code: flatMap(Some(2), x => map(Some(3), y => x + y))
- Meaning:
  - Start with Some(2).
  - flatMap “unboxes” x if present, then applies the provided function f.
  - Inside, map takes Some(3), applies (y => x + y), yielding Some(x + 3).
  - If the outer Option were None, the function wouldn’t run and the result would be None.

- Result here:
  - x = 2; y = 3 → x + y = 5 → Some(5).

2. What “Monad” means (informal)
- A type constructor M<_> with flatMap/bind to chain computations that return M:
  - flatMap: (A -> M<B>) -> M<A> -> M<B>
- For Option:
  - M = Option
  - flatMap lets the next step depend on the value from the previous step and may short‑circuit on emptiness.

3. Why flatMap (vs map) here
- map: A -> B (wraps B back into Option automatically).
- flatMap: A -> Option<B> (you already return an Option; flatMap prevents nesting Option<Option<B>>).
- Pattern:
  - optA.flatMap(a -> optB.map(b -> combine(a,b)))

4. Behavior summary
- Present case:
  - Some(a).flatMap(f) == f(a)
- Empty case:
  - None.flatMap(f) == None
- Short‑circuiting:
  - If any step is None, the rest of the chain is skipped.

5. Sequencing dependent steps (typical shapes)
- Two steps:
  - optA.flatMap(a => optB.map(b => f(a,b)))
- Three steps:
  - optA.flatMap(a => optB.flatMap(b => optC.map(c => g(a,b,c))))

6. Analogy with Promise
- Promise.resolve(2).then(x => Promise.resolve(3).then(y => x+y))
- then is flatMap for Promise: it chains computations that return Promises.
- Same monadic sequencing idea, but async.

7. Purity and referential transparency
- If the functions passed to flatMap/map are pure:
  - Same inputs → same outputs; no side effects.
- Side effects run only when the wrapped values exist.

8. Monad laws (for predictable refactoring)
- Left identity:  flatMap(Some(a), f) == f(a)
- Right identity: flatMap(m, Some) == m
- Associativity:  flatMap(flatMap(m,f), g) == flatMap(m, x => flatMap(f(x), g))
*/

// 9) Natural Transformation — Array<A> -> Option<A>
const headOption = <A>(xs: A[]): Option<A> => xs.length ? Some(xs[0]) : None;
/*
1. Line: a natural transformation between containers (Array -> Option)

- Code: headOption(xs)
- Meaning:
  - Converts an Array<A> into an Option<A> by taking its first element if it exists.
  - If the array is empty, returns None; otherwise, wraps the first element in Some(...).

- Result examples:
  - headOption([10, 20])   → Some(10)
  - headOption([])         → None

2. What "Natural Transformation" means (informal)
- A uniform, structure-preserving mapping between type constructors F and G:
  - For all A, a function nat: F<A> -> G<A> that does not depend on the contents of A.
- Here:
  - F = Array, G = Option, nat = headOption
  - Uniform: same logic works for numbers, strings, objects, etc.

3. Why this is useful
- Changes the context but not the value type:
  - Collapses "many (possibly zero)" (Array) into "zero or one" (Option).
- Useful for composing APIs that expect Option without reshaping values.

4. Behavior and properties
- For any Array<A>:
  - Empty → None
  - Non-empty → Some(first element)
- Pure and referentially transparent:
  - Same input array always yields the same Option, no side effects.

5. Edge cases and tips
- Arrays with `undefined` or `null` values:
  - headOption([undefined]) → Some(undefined)
  - If you want safety, wrap via SomeNullable to skip null/undefined.
- Performance:
  - O(1) time/O(1) space; only checks length and reads index 0.

6. Variations
- lastOption:
  - xs.length ? Some(xs[xs.length-1]) : None
- safeHead for strings:
  - str.length ? Some(str[0]) : None
- headOption with Iterables:
  - Use iterator.next().

7. Composition intuition
- Natural transformations compose:
  - Example: Option<A> -> Either<E,A>
  - Composing with headOption yields Array<A> -> Either<E,A>.
  - Still type-uniform and structure-preserving.
*/

// 10) Monoid — reduce with associative op and identity
const mSum = [1,2,3].reduce((a,b)=>a+b, 0);
const mStr = ['a','b','c'].reduce((a,b)=>a+b, '');
/*
1. Line: reduce with a Monoid (operation + identity)

- Code:
  - [1,2,3].reduce((a,b) => a+b, 0)
  - ['a','b','c'].reduce((a,b) => a+b, '')
- Meaning:
  - A Monoid is a pair (⊕, e) with:
    - Associative binary operation ⊕: (x ⊕ y) ⊕ z == x ⊕ (y ⊕ z)
    - Identity element e: e ⊕ x == x == x ⊕ e
  - For numbers under addition: (⊕ = +, e = 0)
  - For strings under concatenation: (⊕ = +, e = '')

- Why reduce needs a Monoid:
  - Folding (reducing) an array requires an operation that can safely combine elements in any grouping (associativity) and a neutral starting value (identity).

2. Behavior (step-by-step)

- mSum:
  - Start acc = 0
  - 0 + 1 = 1
  - 1 + 2 = 3
  - 3 + 3 = 6
  - Result: 6

- mStr:
  - Start acc = ''
  - '' + 'a' = 'a'
  - 'a' + 'b' = 'ab'
  - 'ab' + 'c' = 'abc'
  - Result: 'abc'

3. Why associativity and identity matter

- Associativity:
  - JavaScript’s Array.reduce may regroup operations internally, especially in parallel contexts (e.g., other languages); associativity ensures correctness.
  - Example: (('a' + 'b') + 'c') == ('a' + ('b' + 'c')) == 'abc'

- Identity:
  - The seed value must not change the result:
    - 0 + x = x
    - '' + s = s

4. Purity and determinism

- With pure operations ((a,b) => a+b), reduce is deterministic:
  - Same array → same output; no side effects.

5. Edge cases and tips

- Empty array:
  - reduce(identity, op) returns the identity (0 or '').
- Non-associative ops:
  - Avoid using operations like subtraction or floating-point addition with sensitive order-dependence.
- Performance:
  - String concatenation is O(n) per step, but manageable for small arrays.

6. Custom Monoids (examples)

- Product monoid (numbers):
  - op = (a,b) => a * b, identity = 1
- Max/Min monoids:
  - op = Math.max, identity = -Infinity
  - op = Math.min, identity = Infinity

7. Composition intuition

- Monoid instances compose:
  - Can define monoids for tuples, objects, etc., by combining underlying monoids.
*/

// 11) ADTs & Pattern Matching
type Shape = { tag: 'Circle'; r: number } | { tag: 'Rect'; w: number; h: number };
const area = (s: Shape): number => s.tag === 'Circle' ? Math.PI * s.r * s.r : s.w * s.h;
/*
1. Line 1: define an Algebraic Data Type (ADT) using a tagged union

- Code:
  - type Shape = { tag: 'Circle'; r: number } | { tag: 'Rect'; w: number; h: number };
- Meaning:
  - Shape is a discriminated union of two object types.
  - The `tag` field is a discriminator identifying the variant.
  - Circle variant has field r (radius); Rect variant has fields w, h.
  - This creates a closed set of variants: Shape = Circle | Rect.

- Why the tag matters:
  - The `tag` acts like a sealed type indicator.
  - The compiler uses it for type narrowing in conditional checks.
  - This prevents confusion when different variants have overlapping field names.

2. Lines 3–4: conditional logic (pattern match via tag check)

- Code:
  - s.tag === 'Circle' ? Math.PI * s.r * s.r : s.w * s.h
- Meaning:
  - If the tag is 'Circle', destructure radius r and compute area.
  - Otherwise, assume Rect: use width and height.
  - TypeScript’s control‑flow analysis narrows type based on `tag`.

3. Behavior (step‑by‑step)

- If s = { tag: 'Circle', r: 2 }:
  - area = π * 2 * 2
- If s = { tag: 'Rect', w: 3, h: 4 }:
  - area = 3 * 4

4. Why ADTs + pattern matching are powerful in TS

- Clarity:
  - Domain models as discriminated unions are explicit and self‑documenting.
- Safety:
  - Type narrowing ensures only valid fields are accessed.
- Immutability:
  - Object literals are often treated as immutable in functional code style.
- Refactoring:
  - Adding a new Shape variant forces compiler errors where tag checks occur.

5. Purity and reasoning

- area is pure:
  - No mutation, no side effects; same input returns same output.
- Referential transparency:
  - Example: area({ tag: 'Circle', r: 2 }) = 12.566... can replace call directly.

6. Edge cases and tips

- Negative/NaN parameters:
  - Not validated—consider checks when constructing values.
- Extensibility:
  - To add Triangle, extend union and update all tag checks.
- Performance:
  - Tag check is O(1). Union types are erased at runtime, so overhead is minimal.
*/

// 12) Effects at the Edges — pure core + I/O boundary
const pureLogic = (x: number) => x * 2;
function mainIO(){ console.log(pureLogic(5)); }
mainIO();
/*
1. Lines: separate pure core from I/O

- Code:
  - pureLogic: a pure function (deterministic, no I/O, no mutation).
  - mainIO: performs side effect (console.log) at the program boundary.
  - mainIO invocation: executes effect after computing pure value.
- Meaning:
  - Keep computation (pureLogic) side‑effect free.
  - Perform side effects (I/O) only at boundaries.

2. Why this separation matters

- Testability:
  - pureLogic is trivial to test: same input → same output.
- Reasoning & refactoring:
  - Pure code is referentially transparent and safe to refactor.
- Reuse & composition:
  - Pure functions can be combined and reused without concern for hidden effects.
- Reliability & observability:
  - I/O centralized in mainIO makes logging/monitoring easier.

3. Behavior (step‑by‑step)

- pureLogic(5) → 10 (pure computation).
- mainIO calls console.log(10) → prints "10" (side effect).

4. Purity vs effects

- pureLogic:
  - No global state, no I/O; returns x * 2 deterministically.
- console.log:
  - Writes to stdout; introduces side effects, not referentially transparent.

5. Edge cases and tips

- Exceptions:
  - Catch errors around I/O boundaries; keep core pure.
- Time, randomness, config:
  - Inject as parameters instead of reading inside pure functions.
- Undefined/NaN:
  - Validate at boundaries before passing to pure functions.

6. Testing strategy

- Unit tests:
  - expect(pureLogic(5)).toBe(10);
- Integration tests:
  - Capture console output from mainIO and assert expected string.

7. Refactoring patterns

- Dependency injection at edges:
  - Pass in a logging function to mainIO instead of using console.log directly.
- Functional style:
  - Core returns data; the boundary decides how to handle it (log, save, send).
*/

// 13) Property‑Based Testing (outline with fast-check)
// import * as fc from 'fast-check';
// fc.assert(fc.property(fc.array(fc.integer()), xs => xs.map(x=>x).every((v,i) => v === xs[i])));
/*
1. Line: a property, not an example — test many random inputs

- Code:
  - fc.property(fc.array(fc.integer()), xs => ... )
  - fc.assert(...) to run it
- Meaning:
  - Property-based test: fast-check generates many random arrays of integers.
  - The assertion must hold for all generated inputs, not just a few fixed examples.

2. The property being checked (Functor identity law for map)

- Law:
  - xs.map(identity) == xs
- In TypeScript:
  - xs.map(x => x) should equal xs element-by-element.
- Intuition:
  - Mapping with the identity function does not change values or structure.

3. Why property‑based tests are powerful

- Broad input coverage:
  - fast-check tries many arrays: empty, large, negatives, duplicates.
- Less bias:
  - You describe the law; the tool generates cases automatically.
- Shrinking:
  - On failure, fast-check reduces the array to a simpler counterexample.

4. Determinism and purity

- Tests are reproducible with a seed.
- Pure transformations (map with identity) are ideal candidates.

5. Extending the idea (other fundamental laws)

- Functor composition:
  - xs.map(x => f(g(x))) == xs.map(g).map(f)
- Monoid laws (reduce):
  - Identity element, associativity checks.
- Option/Optional laws:
  - map(identity) == self
  - map(f).map(g) == map(x => g(f(x)))

6. Edge cases

- Empty arrays: still equal.
- Large arrays: performance concerns.
- Equality semantics: use strict equality by index.

7. Minimal runnable shape

- fc.assert(fc.property(fc.array(fc.integer()), xs =>
-   xs.map(x=>x).every((v,i)=> v===xs[i])
- ))

8. What a failure would mean

- If it fails:
  - Identity map changed something (unexpected) or comparison is wrong.
  - fast-check shows the smallest failing input for debugging.
*/
