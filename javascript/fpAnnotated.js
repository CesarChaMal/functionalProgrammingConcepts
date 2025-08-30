/*
 FP CONCEPTS (JavaScript) — COMPREHENSIVE ANNOTATED GUIDE
 
 CORE CONCEPTS (1-8):
 1) Lambda, Application, Currying, Partial Application — Anonymous functions, function invocation, transforming multi-parameter functions into single-parameter chains
 2) Composition (∘) — Combining functions where output of one becomes input of another
 3) Referential Transparency — Expressions can be replaced with their values without changing program behavior
 4) Immutability — Data structures that cannot be modified after creation
 5) Higher‑Order Functions (HOFs) — Functions that take other functions as parameters or return functions
 6) Functor (map) — Containers that can apply functions to wrapped values while preserving structure
 7) Applicative (ap / mapN) — Enhanced functors that can apply wrapped functions to wrapped values
 8) Monad (flatMap / bind) — Containers supporting sequential computation with context-aware chaining
 
 ADVANCED CONCEPTS (9-16):
 9) Natural Transformation — Structure-preserving mappings between functors
 10) Monoid (associative op + identity) — Types with associative binary operation and identity element
 11) Algebraic Data Types (ADTs) & Pattern Matching — Sum and product types with exhaustive case analysis
 12) Effects at the Edges — Isolating side effects to program boundaries while keeping core logic pure
 13) Property‑Based Testing of Laws — Verifying mathematical properties hold across generated test cases
 14) Lazy Evaluation — Generators, async/await, and deferred computation
 15) Tail Call Optimization — Stack-safe recursion through iteration and trampolines
 16) Advanced Features — Closures, prototypes, and functional patterns
*/

// 1) Lambda, Application, Currying, Partial Application
const inc = x => x + 1;
const add = x => y => x + y;                      // curried
const add5 = add(5);                              // partial
const seven = add5(2);                            // 7
/*
1. Line 1: a simple lambda (a function value)

- Code: const inc = x => x + 1
- Meaning:
  - inc is a function from number to number.
  - It's an arrow function (lambda) that returns x + 1.

- Usage:
  - inc(10) === 11

- Notes:
  - Pure function: same input → same output, no side effects.

2. Line 2: a curried function (a function that returns a function)

- Code: const add = x => y => x + y
- Meaning:
  - add takes one argument x and returns a new function that takes y and computes x + y.
  - This is currying: (x, y) => x + y becomes x => (y => x + y).

- Why this is useful:
  - Enables partial application (fix some args now, pass the rest later).
  - Composes nicely with other higher‑order functions.

- Closure:
  - The inner function (y => x + y) "remembers" x via closure.

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
  - Uncurried version would be const add2 = (x, y) => x + y
  - Currying lets you pre-fill arguments and treat the result as a first‑class function.
*/

// 2) Composition (∘)
const compose = (f, g) => a => f(g(a));
const double = x => x * 2;
const compRes = compose(inc, double)(10);         // 21
/*
1. Line 1: define a generic compose

- Code: const compose = (f, g) => a => f(g(a))
- Meaning:
  - Given g: A -> B and f: B -> C, compose returns a new function h: A -> C such that h(a) = f(g(a)).
  - It builds a pipeline where g runs first, then f.

2. Line 2: define a "double" function

- Code: const double = x => x * 2
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
  - inc is defined (e.g., const inc = x => x + 1).

Extra notes
- Order matters:
  - compose(f, g)(x) = f(g(x)) means "apply g first, then f".

- Mental model:
  - Composition wires small functions together so the output of one feeds the next, without intermediate variables.
*/

// 3) Referential Transparency
const pureTotal = [1,2,3].reduce((a,b)=>a+b, 0); // == 6
// console.log("hi"); // I/O → not referentially transparent
/*
1. What "referential transparency" means

- An expression is referentially transparent if you can replace it with its value everywhere in the program without changing the program's behavior.
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

- Why it's referentially transparent:
  - No I/O, no mutation, no randomness.
  - You can safely replace [1,2,3].reduce((a,b)=>a+b, 0) with 6 anywhere; behavior won't change.

- Therefore: pureTotal is 6; writing const pureTotal = 6 is behaviorally equivalent.

3. The side‑effecting expression (not referentially transparent)

- Code: console.log("hi")
- Why it's not referentially transparent:
  - It performs I/O by printing to stdout.
  - Replacing it with its "value" (undefined) would remove the print, changing observable behavior.
  - So it cannot be freely substituted by a value without altering the program.

Key takeaways
- Pure expressions are interchangeable with their values—great for reasoning, testing, and refactoring.
- Side effects (like printing) are not referentially transparent; they affect the outside world and cannot be replaced by a value without changing behavior.
*/

// 4) Immutability
const p1 = Object.freeze({ x: 1, y: 1 });
const p2 = { ...p1, x: 2 };                      // new object; p1 unchanged
/*
1. Line 1: define an immutable data carrier (using Object.freeze)

- Code: const p1 = Object.freeze({ x: 1, y: 1 });
- What Object.freeze does:
  - Creates an object where all properties are read‑only at runtime.
  - Prevents assignments like p1.x = ... (throws error in strict mode).
  - Works with any object; freezes the immediate properties (shallow freeze).

- Immutability here:
  - JavaScript runtime forbids property reassignment on frozen objects.
  - Note: this is shallow; nested objects need separate freezing for deep immutability.

2. Line 2: create a new instance instead of mutating

- Code: const p2 = { ...p1, x: 2 };
- Step‑by‑step:
  1. Spread p1 to copy its properties into a new object literal (shallow copy).
  2. Override x to 2 while leaving y as p1.y (1).
  3. The result is a new object { x: 2, y: 1 } assigned to p2.

- Key idea:
  - You didn't "change" p1; you built a new value p2 with a different x.
  - p1 remains { x: 1, y: 1 }. p2 is { x: 2, y: 1 }.

Why this matters in FP
- Immutability avoids accidental shared‑state bugs.
- Values are safe to pass around and reuse; reasoning is simpler (no hidden changes).
- Plays well with concurrency and caching.
- When you need a "modified" version, build a new value via spreads (as shown with p2), rather than mutating in place.
*/

// 5) Higher‑Order Functions
const hofSum = [1, 2, 3]
    .map(x => x * 2)
    .filter(x => x > 2)
    .reduce((a, b) => a + b, 0);                  // 10
/*
1. Built-in HOF pipeline (map → filter → reduce)
- What happens:
  - Start with [1, 2, 3]
  - map(x => x*2)       → [2, 4, 6]
  - filter(x => x > 2)  → [4, 6]
  - reduce((a,b)=>a+b) with 0 → 0 + 4 + 6 = 10

- Why these are HOFs:
  - map, filter, reduce each take functions (lambdas) as arguments, so they're higher‑order functions.
- Complexity:
  - Each stage is linear; in practice you traverse the array multiple times (once per stage).
- Purity note:
  - If the lambdas are pure, the pipeline is pure and deterministic.
*/

// Custom HOF #1: map for Array (re-implement to show the idea)
function mapArray(xs, f) {
    const out = new Array(xs.length);
    for (let i = 0; i < xs.length; i++) out[i] = f(xs[i]);
    return out;
}
// Usage: doubles each element -> [2, 4, 6]
const customDoubled = mapArray([1, 2, 3], x => x * 2);
/*
2. Custom HOF #1: mapArray

- Purpose:
  - Transform each element independently using a provided function, producing a new array of possibly different element type.

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

- Why it's higher‑order:
  - It takes a function f as an argument.

- Edge cases:
  - Empty input → returns [].
  - Avoid side effects in f for better reasoning and testability.
*/

// Custom HOF #2: map -> filter -> reduce as one reusable helper
function mapFilterReduce(xs, mapper, predicate, zero, reducer) {
    let acc = zero;
    for (const a of xs) {
        const m = mapper(a);
        if (predicate(m)) acc = reducer(acc, m);
    }
    return acc;
}
// Usage: same logic as the pipeline above -> 10
const customMfr = mapFilterReduce(
    [1, 2, 3],
    x => x * 2,      // [2, 4, 6]
    x => x > 2,      // [4, 6]
    0,
    (a, b) => a + b  // 10
);
/*
3. Custom HOF #2: mapFilterReduce (one reusable helper)

- Purpose:
  - Perform a "map → filter → reduce" pipeline in a single pass, avoiding intermediate arrays.

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

- Why it's higher‑order:
  - Accepts three behaviors (mapper, predicate, reducer).

- When to use:
  - You need the clarity of a pipeline with the efficiency of one pass.
*/

// Custom HOF #3: return a new function by repeating an operation n times
function repeat(n, f) {
    return t => {
        let r = t;
        for (let i = 0; i < n; i++) r = f(r);
        return r;
    };
}
// Usage: reuse an `inc` function, apply it 3 times
const inc3 = repeat(3, inc);
const inc3Res = inc3(10); // 13
/*
4. Custom HOF #3: repeat (returns a new function)

- Purpose:
  - Build a new function that applies f to its input n times:
    repeat(n, f)(x) = f(f(...f(x)...)) with n applications.

- Behavior (step-by-step):
  - Return a function that:
    - Starts r = t
    - Loops i in [0..n): r = f(r)
    - Returns r

- Examples:
  - inc = x => x + 1; repeat(3, inc)(10) → 13
  - exclaim = s => s + "!"; repeat(3, exclaim)("go") → "go!!!"

- Why it's higher‑order:
  - Takes a function and returns a new function.

- Edge cases:
  - n = 0 → identity function (returns input unchanged).
*/

// Option helpers (for 6–9)
const Some = value => ({ _tag: 'Some', value });
const None = { _tag: 'None' };
const isSome = o => o._tag === 'Some';

// 6) Functor: map preserves structure
const map = (oa, f) => isSome(oa) ? Some(f(oa.value)) : None;
const fArr = [1,2,3].map(x => x+1);               // [2,3,4]
const fOpt = map(Some(42), x => x + 1);           // Some(43)
/*
Explanation of Functor behavior in JavaScript map

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
   - For Option: eliminates manual null/undefined checks.

4. Behavior summary:
   - Array:
       [1,2,3].map(x => x+1) = [2,3,4]
   - Option:
       map(Some(42), x => x+1) = Some(43)
       map(None, x => x+1)     = None

5. Functor laws:
   - Identity:       map(fa, x => x) == fa
   - Composition:    map(fa, x => g(f(x))) == map(map(fa,f), g)
*/

// 7) Applicative: ap / liftA2 combine independent contexts
const ap = (of, oa) => isSome(of) && isSome(oa) ? Some(of.value(oa.value)) : None;
const pure = x => Some(x);
const liftA2 = f => oa => ob => ap(ap(pure(a => b => f(a,b)), oa), ob);
const name = Some('Ada');
const age = Some(36);
const user = liftA2((n, a) => ({ n, a }))(name)(age); // Some({n:'Ada',a:36})
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

3. Step-by-step evaluation of user:
- pure(a => b => f(a,b)) lifts a curried function into Option.
- ap(..., name) applies it to Some("Ada") → Some(b => f("Ada", b)).
- ap(..., age) applies that to Some(36) → Some(f("Ada",36)).
- f("Ada",36) = {n:"Ada", a:36}.
- Final result: Some({n:"Ada",a:36}).

4. Use cases:
- Building objects from multiple Option values.
- Safe combination of independent inputs without null-checks.
*/

// 8) Monad: flatMap for dependent sequencing
const flatMap = (oa, f) => isSome(oa) ? f(oa.value) : None;
const monadRes = flatMap(Some(2), x => map(Some(3), y => x + y));     // Some(5)
/*
1. Line: flatMap sequences dependent computations (Monad behavior)

- Code: flatMap(Some(2), x => map(Some(3), y => x + y))
- Meaning:
  - Start with Some(2).
  - flatMap "unboxes" x if present, then applies the provided function f.
  - Inside, map takes Some(3), applies (y => x + y), yielding Some(x + 3).
  - If the outer Option were None, the function wouldn't run and the result would be None.

- Result here:
  - x = 2; y = 3 → x + y = 5 → Some(5).

2. What "Monad" means (informal)
- A type constructor M<_> with flatMap/bind to chain computations that return M:
  - flatMap: (A -> M<B>) -> M<A> -> M<B>
- For Option:
  - M = Option
  - flatMap lets the next step depend on the value from the previous step and may short‑circuit on emptiness.

3. Why flatMap (vs map) here
- map: A -> B (wraps B back into Option automatically).
- flatMap: A -> Option<B> (you already return an Option; flatMap prevents nesting Option<Option<B>>).

4. Behavior summary
- Present case:
  - Some(a).flatMap(f) == f(a)
- Empty case:
  - None.flatMap(f) == None
- Short‑circuiting:
  - If any step is None, the rest of the chain is skipped.

5. Monad laws (for predictable refactoring)
- Left identity:  flatMap(Some(a), f) == f(a)
- Right identity: flatMap(m, Some) == m
- Associativity:  flatMap(flatMap(m,f), g) == flatMap(m, x => flatMap(f(x), g))
*/

// 9) Natural Transformation — Array -> Option
const headOption = xs => xs.length ? Some(xs[0]) : None;
/*
1. Line: a natural transformation between containers (Array -> Option)

- Code: headOption(xs)
- Meaning:
  - Converts an Array into an Option by taking its first element if it exists.
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
  - JavaScript's Array.reduce may regroup operations internally; associativity ensures correctness.
  - Example: (('a' + 'b') + 'c') == ('a' + ('b' + 'c')) == 'abc'

- Identity:
  - The seed value must not change the result:
    - 0 + x = x
    - '' + s = s
*/

// 11) ADTs & Pattern Matching
const Circle = r => ({ tag: 'Circle', r });
const Rect = (w, h) => ({ tag: 'Rect', w, h });
const area = s => s.tag === 'Circle' ? Math.PI * s.r * s.r : s.w * s.h;
/*
1. Line 1-2: define an Algebraic Data Type (ADT) using tagged objects

- Code:
  - const Circle = r => ({ tag: 'Circle', r });
  - const Rect = (w, h) => ({ tag: 'Rect', w, h });
- Meaning:
  - Shape is a discriminated union of two object types.
  - The `tag` field is a discriminator identifying the variant.
  - Circle variant has field r (radius); Rect variant has fields w, h.
  - This creates a closed set of variants: Shape = Circle | Rect.

- Why the tag matters:
  - The `tag` acts like a sealed type indicator.
  - Prevents confusion when different variants have overlapping field names.

2. Line 3: conditional logic (pattern match via tag check)

- Code:
  - s.tag === 'Circle' ? Math.PI * s.r * s.r : s.w * s.h
- Meaning:
  - If the tag is 'Circle', destructure radius r and compute area.
  - Otherwise, assume Rect: use width and height.

3. Behavior (step‑by‑step)

- If s = Circle(2):
  - area = π * 2 * 2
- If s = Rect(3, 4):
  - area = 3 * 4

4. Why ADTs + pattern matching are powerful
- Clarity: Domain models as discriminated unions are explicit and self‑documenting.
- Safety: Tag checks ensure only valid fields are accessed.
- Immutability: Object literals are often treated as immutable in functional code style.
*/

// 12) Effects at the Edges — pure core + I/O boundary
const pureLogic = x => x * 2;
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

3. Behavior (step‑by‑step)

- pureLogic(5) → 10 (pure computation).
- mainIO calls console.log(10) → prints "10" (side effect).

4. Testing strategy

- Unit tests:
  - expect(pureLogic(5)).toBe(10);
- Integration tests:
  - Capture console output from mainIO and assert expected string.
*/

// 13) Property‑Based Testing (outline)
// Example: test that map with identity preserves array
const testMapIdentity = xs => {
    const mapped = xs.map(x => x);
    return mapped.every((v, i) => v === xs[i]);
};
/*
1. Property-based test: test many random inputs

- Code:
  - testMapIdentity checks xs.map(x => x) === xs element-by-element
- Meaning:
  - Property-based test: generates many random arrays.
  - The assertion must hold for all generated inputs, not just a few fixed examples.

2. The property being checked (Functor identity law for map)

- Law:
  - xs.map(identity) == xs
- In JavaScript:
  - xs.map(x => x) should equal xs element-by-element.
- Intuition:
  - Mapping with the identity function does not change values or structure.

3. Why property‑based tests are powerful

- Broad input coverage:
  - Tries many arrays: empty, large, negatives, duplicates.
- Less bias:
  - You describe the law; the tool generates cases automatically.
*/

// 14) Lazy Evaluation: generators and async patterns
function* fibonacci() {
    let [a, b] = [0, 1];
    while (true) {
        yield a;
        [a, b] = [b, a + b];
    }
}

// Only computes values as needed
const first10Fibs = Array.from({ length: 10 }, (_, i) => {
    const gen = fibonacci();
    for (let j = 0; j < i; j++) gen.next();
    return gen.next().value;
}); // [0,1,1,2,3,5,8,13,21,34]

// Lazy property with getter
class LazyValue {
    constructor(computation) {
        this.computation = computation;
        this.computed = false;
        this.value = undefined;
    }
    
    get() {
        if (!this.computed) {
            console.log('Computing...');
            this.value = this.computation();
            this.computed = true;
        }
        return this.value;
    }
}

const lazyResult = new LazyValue(() => 42);

// 15) Tail Call Optimization: iteration and trampolines
function factorialIterative(n) {
    let result = 1;
    while (n > 1) {
        result *= n--;
    }
    return result;
}

// Trampoline pattern
function trampoline(fn) {
    let result = fn();
    while (typeof result === 'function') {
        result = result();
    }
    return result;
}

function factorialTrampoline(n, acc = 1) {
    if (n <= 1) return acc;
    return () => factorialTrampoline(n - 1, n * acc);
}

const fact5 = trampoline(() => factorialTrampoline(5)); // 120

// 16) Advanced JavaScript Features
const memoize = fn => {
    const cache = new Map();
    return (...args) => {
        const key = JSON.stringify(args);
        if (cache.has(key)) return cache.get(key);
        const result = fn(...args);
        cache.set(key, result);
        return result;
    };
};

const fibonacci2 = memoize(n => {
    if (n <= 1) return n;
    return fibonacci2(n - 1) + fibonacci2(n - 2);
});

const fib10 = fibonacci2(10); // 55

// === DEMONSTRATIONS ===

console.log('=== BASIC CONCEPTS ===');
console.log('Currying:', seven);
console.log('Composition:', compRes);
console.log('HOF Pipeline:', hofSum);
console.log('Custom HOFs:', customDoubled, customMfr, inc3Res);

console.log('\n=== FUNCTORS & MONADS ===');
console.log('Array Functor:', fArr);
console.log('Option Functor:', fOpt);
console.log('Applicative:', user);
console.log('Monad:', monadRes);

console.log('\n=== MONOIDS & ADTS ===');
console.log('Sum Monoid:', mSum);
console.log('String Monoid:', mStr);
console.log('Circle Area:', area(Circle(2)));
console.log('Rect Area:', area(Rect(3, 4)));

console.log('\n=== ADVANCED ===');
console.log('Head Option:', headOption([1, 2, 3]), headOption([]));
console.log('Lazy Fibonacci:', first10Fibs.slice(0, 5));
console.log('TCO Factorial:', fact5);
console.log('Memoized Fibonacci:', fib10);
console.log('Map Identity Test:', testMapIdentity([1, 2, 3]));

/*
REFERENCE MAPPING (JavaScript):

| Concept                   | JavaScript Implementation                     |
|---------------------------|-----------------------------------------------|
| Lambda                    | const f = x => x + 1                         |
| Currying                  | const add = x => y => x + y                   |
| Composition               | const compose = (f, g) => x => f(g(x))        |
| Immutability              | Object.freeze(), spread operator             |
| Higher-Order Functions    | map, filter, reduce                           |
| Functor                   | map function preserving structure             |
| Applicative               | ap function for independent contexts          |
| Monad                     | flatMap for dependent sequencing             |
| Pattern Matching          | tag-based conditional logic                   |
| Lazy Evaluation           | Generator functions (function*)               |
| Tail Call Optimization    | Trampoline pattern                            |
| Memoization               | Closure with Map cache                        |
*/