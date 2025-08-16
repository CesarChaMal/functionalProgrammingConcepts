/*
 FP CONCEPTS (Java 17+) — ALL IN ONE, ANNOTATED
 1) Lambda, Application, Currying, Partial Application — Lambdas + Function composition.
 2) Composition (∘) — Use Function#compose/andThen.
 3) Referential Transparency — Pure expressions can be substituted by values.
 4) Immutability — Records are concise immutable carriers.
 5) Higher‑Order Functions — Stream pipelines with map/filter/reduce.
 6) Functor — Optional.map.
 7) Applicative — map2 to combine independent Optionals.
 8) Monad — Optional.flatMap for dependent sequencing.
 9) Natural Transformation — List<A> -> Optional<A>.
 10) Monoid — Reduce with associative op and identity.
 11) ADTs & Pattern Matching — sealed interfaces + switch.
 12) Effects at the Edges — Keep core pure; do I/O in main.
 13) Property‑Based Testing — Outline (jqwik).
*/

import java.util.*;
import java.util.function.*;

public class FPJavaAnnotated {
  // 1) Lambda, Application, Currying, Partial Application
  static final Function<Integer,Integer> inc = x -> x + 1;                      // λx. x+1
  static final Function<Integer,Function<Integer,Integer>> add = x -> y -> x + y; // curried
  // add.apply(5) → returns function f(y) = 5 + y
  // add5 = f
  static final Function<Integer,Integer> add5 = add.apply(5);                   // partial
  // add5.apply(2) → f(2) = 5 + 2 = 7
  // seven = 7
  static final int seven = add5.apply(2);                                       // 7
/*
1. Line 1: a simple lambda (a function value)

- Code: Function<Integer,Integer> inc = x -> x + 1;
- What it means:
            - Function<Integer, Integer> is a Java standard functional interface whose single method is apply(Integer a).
            - The lambda x -> x + 1 is an anonymous function: it takes one parameter x (of type Integer) and returns x + 1 (also an Integer).
            - You can think of inc as a variable that holds a function. It’s a value just like a number or string, but callable.

            - How you use it:
            - inc.apply(10) returns 11.

            - Key ideas:
            - Lambda: a concise way to write a function inline.
            - Type: Input Integer -> Output Integer.
            - Purity: Given the same input, it always returns the same output and has no side effects.

2. Line 2: a curried function (a function that returns a function)

- Code: Function<Integer,Function<Integer,Integer>> add = x -> y -> x + y;
- What it means:
            - The type says: given an Integer, return a Function<Integer, Integer>.
            - The lambda x -> y -> x + y is “curried” form: it does not take two arguments at once. Instead:
            - First call: give it x; it returns another function that “remembers” x.
        - Second call: give that returned function y; it computes x + y.

- Why this is powerful:
            - It enables partial application—fixing some arguments early and producing specialized functions.
            - It also composes nicely with other higher-order functions.

            - Step-by-step mental model:
            - add.apply(5) produces a new function f such that f.apply(y) = 5 + y.
    - That’s because the inner y -> x + y closes over x, keeping its value via a closure.

- Equivalence to a two-argument function:
            - An uncurried version would look like BiFunction<Integer,Integer,Integer> add2 = (x,y) -> x + y;
    - Currying transforms (x,y) -> z into x -> (y -> z).

3. Line 3: partial application (fix the first argument)

- Code: Function<Integer,Integer> add5 = add.apply(5);
- What it means:
            - We call add with the first argument 5, but we don’t supply y yet.
            - This returns a new function add5 that expects only one number y and will return 5 + y.
    - Type of add5 is Function<Integer, Integer>.

- What actually happens:
            - add.apply(5) evaluates x -> y -> x + y with x bound to 5 and returns y -> 5 + y.

- How you use it:
            - add5.apply(10) returns 15.

            - Key concept: partial application
    - You provided part of the needed inputs (the first one), and got back a function waiting for the rest.

4. Line 4: function application (call the specialized function)

- Code: int seven = add5.apply(2);
- What it means:
            - We now supply the remaining input y = 2 to the function generated in the previous step.
            - It computes 5 + 2 = 7 and stores it in seven.

- Step-by-step execution flow (all lines together):
            - inc = x -> x + 1
            - add = x -> (y -> x + y)
            - add5 = add.apply(5) = (y -> 5 + y)
            - seven = add5.apply(2) = 7

    Extra notes for beginners
- Boxing vs primitives:
            - Function<Integer, Integer> works with Integer (the boxed type), not int. Java will auto-box/unbox when you use int literals like 5.

            - Closure:
            - When you partially apply add with 5, the returned function “remembers” x = 5. That’s a closure—an inner function capturing variables from its creation environment.

- Why currying is useful:
            - You can create reusable, specialized functions easily.
            - It aligns well with composition and higher-order functions (functions returning or taking functions).

            - Compare curried vs uncurried style
    - Curried:
    Function<Integer, Function<Integer, Integer>> add = x -> y -> x + y;
    Function<Integer, Integer> add10 = add.apply(10);
    int result = add10.apply(3); // 13

    - Uncurried (two args at once):
    BiFunction<Integer, Integer, Integer> add2 = (x, y) -> x + y;
    int result = add2.apply(10, 3); // 13

- Currying lets you pre-fill arguments and treat the result as a first-class function.
*/

  // 2) Composition (∘)
  static final Function<Integer,Integer> dbl = x -> x * 2;
  static final Function<Integer,Integer> dblThenInc = inc.compose(dbl);         // inc(dbl(x))
  static final int compRes = dblThenInc.apply(10);                               // 21
/*
1. Line 1: define a “double” function

- Code: Function<Integer,Integer> dbl = x -> x * 2;
- Meaning:
            - dbl is a function taking an Integer x and returning x * 2.
            - Type-wise: Integer -> Integer.

- Usage:
            - dbl.apply(7) returns 14.

2. Line 2: compose two functions (do dbl first, then inc)

- Code: Function<Integer,Integer> dblThenInc = inc.compose(dbl);
- What compose does:
            - For functions f and g, f.compose(g) builds a new function h such that h(x) = f(g(x)).
            - The right-hand function (g) runs first; then the left one (f) runs on its result.

            - In your case:
            - f is inc (adds 1).
            - g is dbl (multiplies by 2).
            - So dblThenInc(x) = inc(dbl(x)) = (x * 2) + 1.

            - Type:
            - Still Function<Integer, Integer>.

3. Line 3: apply the composed function

- Code: int compRes = dblThenInc.apply(10);
- Step-by-step trace:
            1. Call dblThenInc with 10 (10 is auto-boxed to Integer for apply).
            2. By definition of compose, evaluate dbl first: dbl.apply(10) = 10 * 2 = 20.
            3. Then apply inc to that result: inc.apply(20) = 20 + 1 = 21.
            4. Return 21 (auto-unboxed to int), assign to compRes.

- Final result: compRes == 21.

    Extra notes
- Order matters:
            - inc.compose(dbl) means “double, then increment”.
            - The alternative inc.andThen(dbl) would mean “increment, then double”.

            - Mental model:
            - Composition builds pipelines of small functions. Think “output of one becomes input of the next” without temporary variables.
*/

  // 3) Referential Transparency
  static final int pureTotal = List.of(1,2,3).stream().reduce(0, Integer::sum); // 6
  // System.out.println("hi"); // side effect → not referentially transparent
/*
1. What “referential transparency” means

- An expression is referentially transparent if you can replace it with its value everywhere in the program without changing the program’s behavior.
            - In practice: no side effects, no dependence on hidden/external state, and it always produces the same result for the same inputs.

2. The pure expression

- Code: List.of(1,2,3).stream().reduce(0, Integer::sum)
- Meaning:
            - List.of(1,2,3) creates an immutable list [1, 2, 3].
            - .stream() streams its elements.
            - .reduce(0, Integer::sum) folds the elements starting with initial value 0, using Integer::sum (which is equivalent to (a, b) -> a + b).

            - Micro-trace of reduce:
            - Start: acc = 0
            - Step 1: acc = 0 + 1 = 1
            - Step 2: acc = 1 + 2 = 3
            - Step 3: acc = 3 + 3 = 6
            - Result: 6

            - Why it’s referentially transparent:
            - No I/O, no mutation, no randomness. It’s deterministic and pure.
            - You can safely replace the whole expression with 6 anywhere, and nothing else changes.

- Therefore: pureTotal is 6, and the program behaves the same if you write static final int pureTotal = 6;

3. The side-effecting expression (not referentially transparent)

- Code: System.out.println("hi");
- Why it’s not referentially transparent:
            - It performs I/O by writing to stdout.
    - If you replace this expression with its “value” (there is none; it returns void), you lose the printed output, which changes the program’s observable behavior.
    - Thus, you cannot freely substitute it with a value without altering behavior.

            Key takeaways
- Pure expressions (like the reduce example) are interchangeable with their values—great for reasoning, testing, and refactoring.
            - Side effects (like printing) are not referentially transparent; they affect the outside world and can’t be replaced by a value without changing behavior.
*/

  // 4) Immutability via records
  public record Point(int x, int y) {}
  static final Point p1 = new Point(1,1);
  static final Point p2 = new Point(2, p1.y()); // new instance; p1 unchanged
/*
1. Line 1: define an immutable data carrier (a record)

- Code: public record Point(int x, int y) {}
- What a record is:
            - A compact class for “data with value semantics”.
            - Automatically generates:
            - A canonical constructor Point(int x, int y)
        - Accessors x() and y() (not getX/getY)
            - equals, hashCode, and toString

    - Components (x, y) are final; you cannot reassign them after construction.

            - Immutability here:
            - You can’t do p.x = ... or p.y = ...; records don’t expose setters.
    - Note: immutability is shallow—if a component is a mutable object, the record won’t “freeze” it. Using primitives (int) here is fully safe.

2. Line 2: create the first instance

- Code: static final Point p1 = new Point(1,1);
- Meaning:
            - Calls the generated constructor with x = 1, y = 1.
            - p1 is a constant reference to an immutable Point(1, 1).
            - Accessors: p1.x() returns 1, p1.y() returns 1.

3. Line 3: create a new instance instead of mutating

- Code: static final Point p2 = new Point(2, p1.y());
- Step-by-step:
            1. Call p1.y() → returns 1 (reads, not mutates).
            2. Construct a new Point with x = 2 and y = 1 → Point(2, 1).
            3. Assign it to p2.

            - Key idea:
            - You didn’t “change” p1; you built a new value (p2) that shares y with p1 but has a different x.
            - p1 remains Point(1, 1). p2 is Point(2, 1).

    Why this matters in FP
- Immutability avoids accidental shared-state bugs.
            - Values are safe to pass around and reuse; reasoning is simpler (no hidden changes).
            - Works well with parallelism and caching.
            - If you need a “modified” version, you create a new instance with the desired changes (as shown with p2).
*/

    // 5) Higher‑Order Functions
    static final int hofSum = List.of(1,2,3).stream().map(x->x*2).filter(x->x>2).reduce(0, Integer::sum);
/*
1. Built-in HOF pipeline (map → filter → reduce)
- What happens:
            - Start with [1, 2, 3]
            - map(x -> x*2) → [2, 4, 6]
            - filter(x -> x > 2) → [4, 6]
            - reduce(0, sum) → 0 + 4 + 6 = 10

            - Why these are HOFs:
            - map, filter, reduce all receive functions as parameters (lambdas), so they’re higher-order functions.
*/
    // Custom HOF #1: map for List
    static <A,B> List<B> map(List<A> xs, Function<A,B> f) {
        List<B> out = new ArrayList<>(xs.size());
        for (A a : xs) out.add(f.apply(a));
        return out;
    }
    // Usage: doubles each element -> [2,4,6]
    static final List<Integer> customDoubled = map(List.of(1,2,3), x -> x * 2);
/*
2. Custom HOF #1: map for List
- Purpose:
    - Transform each element of a list independently using a provided function, producing a new list of potentially different element type.

- Signature reasoning:
    - map(List[ xs, Function<A,B> f): List]()
    - [**A and B are generic type parameters: A is the input element type, B is the output element type. The function f: A -> B is the “behavior” you pass in.**]()

- [**Behavior (step-by-step): **]()
    - [**Allocate an output list sized like the input (eager, not streaming). **]()
    - [**For each element a in xs: **]()
        - [**Compute b = f.apply(a). **]()
        - [**Append b to out. **]()

    - [**Return out. **]()

- [**Example: **]()
    - [**map(List.of(1,2,3), x -> x * 2) → [2, 4, 6] **]()
    - [**map(List.of("a", "bb"), s -> s.length()) → [1, 2] **]()

- [**Complexity: **]()
    - [**Time O(n), Space O(n) for output, where n = xs.size(). **]()

- [**Purity and referential transparency: **]()
    - [**If f is pure, map is pure; the output depends only on xs and f, with no side effects. **]()

- [**Why it’s higher‑order: **]()
    - [**It takes a function as an argument (f). **]()

- [**Variations: **]()
    - [**You could implement it lazily (e.g., with streams or iterables) to avoid allocating unless consumed. **]()
    - [**Could be parallelized if f is pure and independent per element. **]()

- [**Edge cases: **]()
    - [**Empty input → returns empty output. **]()
    - [**Null elements: allowed only if f can handle null; otherwise avoid nulls or use Optional. **]()

*/
    // Custom HOF #2: map -> filter -> reduce as one reusable helper
    static <A,B> B mapFilterReduce(
            List<A> xs,
            Function<A,A> mapper,
            Predicate<A> predicate,
            B zero,
            BiFunction<B,A,B> reducer
    ) {
        B acc = zero;
        for (A a : xs) {
            A m = mapper.apply(a);
            if (predicate.test(m)) acc = reducer.apply(acc, m);
        }
        return acc;
    }
    // Usage: same logic as the stream pipeline above -> 10
    static final int customMfr = mapFilterReduce(
            List.of(1,2,3),
            x -> x * 2,
            x -> x > 2,
            0,
            Integer::sum
    );
/*
3. [**Custom HOF #2: mapFilterReduce (one reusable helper)**]()
- [**Purpose: **]()
    - [**Perform a “map → filter → reduce” pipeline in one pass over the data. Useful for performance when you want to avoid intermediate lists/streams. **]()

- [**Signature reasoning:**]()** ****[]()**
    - **[mapFilterReduce(List xs, Function<A,A> mapper, Predicate predicate, B zero, BiFunction<B,A,B> reducer): B]() **
    - **[mapper: A -> A keeps the element type consistent through map/filter (simplifies API). You can generalize to mapper: A -> C with more type parameters (then predicate/reducer would work on C).]() **
    - **[zero is the identity/initial accumulator value.]() **
    - **[reducer: (B, A) -> B folds an element of type A into an accumulator of type B.]() **

- **[Behavior (step-by-step): ]()**
    - **Initialize acc = zero. **
    - **For each element a in xs: **
        - **m = mapper.apply(a) // “map step” **
        - **If predicate.test(m): // “filter step” **
            - **acc = reducer.apply(acc, m) // “reduce step” **

    - **Return acc. **

- **[Example (mirrors the stream pipeline): ]()**
    - **xs = [1, 2, 3] **
    - **mapper = x -> x * 2 → [2, 4, 6] **
    - **predicate = x -> x > 2 → keep [4, 6] **
    - **zero = 0 **
    - **reducer = Integer::sum → 0 + 4 + 6 = 10 **
    - **Result: 10 **

- **[Complexity: ]()**
    - **One pass, Time O(n), Space O(1) extra. **

- **[Purity and referential transparency: ]()**
    - **If mapper, predicate, and reducer are pure, this helper is pure. **
    - **Deterministic: same xs and same functions → same result. **

- **[Why it’s higher‑order: ]()**
    - **Accepts three behaviors: mapper, predicate, reducer. **

- **[When to use: ]()**
    - **When you want the clarity of a pipeline but the performance of a single traversal without intermediate collections/streams. **

- **[Variations: ]()**
    - **Generalize mapper: Function<A,C>, predicate: Predicate, reducer: BiFunction**** to allow type changes after mapping. ******
    - **Add short‑circuiting by letting reducer signal early termination (e.g., via a wrapper that can indicate “stop”). **

- **[Edge cases: ]()**
    - **Empty input → returns zero. **
    - **If predicate always false → returns zero. **
    - **If reducer is non‑associative or has side effects, results may be surprising; prefer pure, associative reducers where possible.**

*/

    // Custom HOF #3: return a new function by repeating an operation n times
    static <T> Function<T,T> repeat(int n, Function<T,T> f) {
        return t -> {
            T r = t;
            for (int i = 0; i < n; i++) r = f.apply(r);
            return r;
        };
    }
    // Usage: reuse the existing `inc` function, apply it 3 times
    static final Function<Integer,Integer> inc3 = repeat(3, inc);
    static final int inc3Res = inc3.apply(10); // 13
/*
4. **[Custom HOF #3: repeat (returns a new function)]()**
- **[Purpose: ]()**
    - **Build a new function that applies a given function n times to its input: repeat(n, f)(t) = f(f(...f(t)...)) with n applications. **

- **[Signature reasoning: ]()**
    - **repeat(int n, Function<T,T> f): Function<T,T> **
    - **The function f must be endo on T (input and output both T) so it can be composed with itself. **

- **[Behavior (step-by-step): ]()**
    - **Returns a lambda that: **
        - **Starts with r = t (the input). **
        - **Loops i from 0 to n‑1: **
            - **r = f.apply(r) **

        - **Returns r. **

- **[Example: ]()**
    - **Let inc = x -> x + 1. **
    - **inc3 = repeat(3, inc); inc3.apply(10) → 13. **
    - **For strings: repeat(3, s -> s + "!").apply("go") → "go!!!". **

- **[Complexity: ]()**
    - **Each call to the returned function costs O(n) applications of f. **
    - **The builder itself is O(1). **

- **[Purity and referential transparency: ]()**
    - **If f is pure, repeat returns a pure function; calling it with the same input always yields the same output. **

- **[Why it’s higher‑order: ]()**
    - **Takes a function (f) and returns a new function. **

- **[Variations: ]()**
    - **Precompose f n times into a single function to avoid loop overhead on each call (trades upfront composition for faster application). **
    - **Support n = 0 by returning identity (the provided implementation already behaves that way because the loop doesn’t run). **

- **[Edge cases: ]()**
    - **n = 0 → identity function (returns input unchanged). **
    - **Large n with expensive f → consider performance or algebraic shortcuts (e.g., exponentiation by squaring if f is combinable). **

- **[When to use: ]()**
    - **Reusing a transformation multiple times without writing manual loops. **
    - **Creating reusable “powered” behaviors (e.g., move3Steps = repeat(3, moveOneStep)). **

**[Testing tips and small examples]()**
- **[map: ]()**
    - **Identity law: map(xs, identity) == xs. **
    - **Composition law: map(xs, f ∘ g) == map(map(xs, g), f). **

- **[mapFilterReduce: ]()**
    - **Should match the equivalent separate steps on the same input: **
        - **reduce(map(filter(map(xs, mapper), predicate), identity), zero, reducer) == mapFilterReduce(xs, mapper, predicate, zero, reducer) **
        - **Or compare directly with a stream pipeline for a few sample cases. **

- **[repeat: ]()**
    - **n = 0 → identity: repeat(0, f).apply(x) == x. **
    - **n = 1 → f itself: repeat(1, f).apply(x) == f(x). **
    - **n + m → composition: repeat(n + m, f).apply(x) == repeat(n, f).andThen(repeat(m, f)).apply(x), if you implement a composed variant.**

Key takeaways
    - All these are higher-order because they take functions as parameters or return functions.
    - The stream example uses standard library HOFs; the custom examples show how to build your own reusable functional utilities.
            - mapFilterReduce demonstrates how to package a common pipeline into a single, composable abstraction.
    - repeat shows building new behavior (n-fold application) by function return.
*/

  // 6) Functor (Optional.map)
  static final Optional<Integer> fOpt = Optional.of(42).map(x -> x + 1); // Optional[43]
/*
1. Line: map over Optional (Functor behavior)

- Code: Optional.of(42).map(x -> x + 1)
- Meaning:
  - Optional.of(42) constructs a container that may hold a value (here it does: 42).
  - map applies a function to the contained value if present, wrapping the result back in an Optional.
  - If the Optional is empty, map skips the function and returns Optional.empty().

- Result here:
  - 42 → 42 + 1 = 43 → Optional[43]

2. What “Functor” means (informal)
- A type constructor F<_> that supports a structure‑preserving mapping operation:
  - map: (A -> B) -> F<A> -> F<B>
- For Optional:
  - F = Optional
  - map(f, Optional<A>) → Optional<B>
  - “Preserve structure” means: presence/emptiness is untouched; only the inner value (if any) changes.

3. Why Optional.map is useful
- Eliminates manual presence checks:
  - Instead of:
    if (opt.isPresent()) return Optional.of(f(opt.get()));
    else return Optional.empty();
  - Use: opt.map(f)
- Encourages null‑safe, composable transformations.

4. Behavior summary
- Present case: Optional.of(v).map(f) == Optional.of(f(v))
- Empty case:  Optional.<A>empty().map(f) == Optional.<B>empty()

5. Functor laws (enable safe refactoring)
- Identity:       opt.map(x -> x) == opt
- Composition:    opt.map(x -> g(f(x))) == opt.map(f).map(g)
  - These hold for pure f and g, so you can fuse or reorder maps without changing behavior.

6. Purity and effects
- If the mapper is pure (no side effects), Optional.map is pure and referentially transparent:
  - Same input + same mapper → same output.
- If the mapper has side effects, they execute only when the value is present.

7. Edge cases and tips
- Null safety:
  - Optional.of(null) throws; use Optional.ofNullable(x) when x may be null.
- Cost:
  - O(1): at most one function call and one wrapper allocation.
- map vs flatMap:
  - Use map when f: A -> B (plain value).
  - Use flatMap when f: A -> Optional<B> to avoid nesting Optional<Optional<B>>.

8. Chaining example
- Optional.of(10)
    .map(x -> x + 1)   // Optional[11]
    .map(x -> x * 2)   // Optional[22]
*/

  // 7) Applicative (map2): combine independent Optionals
  static <A,B,C> Optional<C> map2(Optional<A> oa, Optional<B> ob, BiFunction<A,B,C> f) {
    return oa.flatMap(a -> ob.map(b -> f.apply(a,b)));
  }
  static final Optional<String> user = map2(Optional.of("Ada"), Optional.of(36), (n,a) -> n + " is " + a);
/*
1. Line: combine two independent Optionals with map2 (Applicative behavior)

- Code: map2(oa, ob, f) where f: (A, B) -> C
- Meaning:
  - If both Optionals are present, apply f to their inner values and wrap the result.
  - If either is empty, the result is Optional.empty() (no combination happens).

- Result here:
  - Both present: "Ada" and 36 → f("Ada", 36) = "Ada is 36" → Optional["Ada is 36"].

2. Why “Applicative” (intuition)
- Applicative combines independent contexts:
  - “Independent” means neither value depends on the other to exist (no sequencing dependency).
- Signature shape:
  - liftA2/map2: (A, B) -> C, Optional<A>, Optional<B> => Optional<C>
- Contrast with Monad:
  - Monad (flatMap) handles dependent sequencing; Applicative handles parallel/independent combination.

3. How map2 works (step-by-step)
- oa.flatMap(a ->
    ob.map(b ->
      f.apply(a, b)
    )
  )
- If oa is empty → short‑circuit to empty.
- If oa is present but ob is empty → map is skipped → empty.
- If both present → apply f(a, b) and wrap.

4. Common use cases
- Building records/DTOs from multiple Optional fields (e.g., name, age, email).
- Validating and combining independent inputs (presence‑only; for accumulating errors use a Validation type, not Optional).

5. Purity and referential transparency
- If f is pure, map2 is pure and deterministic:
  - Same inputs → same output; no side effects.
- Side effects in f run only when both values are present.

6. Edge cases and tips
- Nulls:
  - Avoid Optional.of(null); use Optional.ofNullable(x) before passing to map2.
- Short‑circuiting:
  - Stops at the first empty; no call to f unless both are present.
- Arity:
  - For 3+ inputs, nest map2 or define map3/mapN helpers:
    - map2(oa, ob, (a,b) -> pair) then map2(pair, oc, (p,c) -> ...), etc.

7. Small variations
- Using method refs:
  - map2(oa, ob, YourClass::combine)
- Building tuples:
  - map2(oa, ob, (a,b) -> new AbstractMap.SimpleEntry<>(a,b))

8. Laws (Applicative intuition, assuming pure functions)
- Homomorphism: map2(Optional.of(a), Optional.of(b), f) == Optional.of(f(a,b))
- Interchange/Composition (informal for Optional): combination is associative up to restructuring when all present; empties dominate.
*/

  // 8) Monad (flatMap): dependent sequencing
  static final Optional<Integer> monadRes = Optional.of(2).flatMap(x -> Optional.of(3).map(y -> x + y));
/*
1. Line: flatMap sequences dependent computations (Monad behavior)

- Code: Optional.of(2).flatMap(x -> Optional.of(3).map(y -> x + y))
- Meaning:
  - Start with Optional[2].
  - flatMap “unboxes” x if present, then runs the next computation which may also be Optional‑producing.
  - Inside, map transforms Optional[3] by adding x, yielding Optional[x + 3].
  - If the outer Optional were empty, the lambda wouldn’t run and the result would be Optional.empty().

- Result here:
  - x = 2; y = 3 → x + y = 5 → Optional[5].

2. What “Monad” means (informal)
- A type constructor M<_> with flatMap/bind to chain computations that return M:
  - flatMap: (A -> M<B>) -> M<A> -> M<B>
- For Optional:
  - M = Optional
  - flatMap lets the next step depend on the value from the previous step and may short‑circuit on emptiness.

3. Why flatMap (vs map) here
- map: A -> B (wraps B back into Optional automatically).
- flatMap: A -> Optional<B> (you already return an Optional; flatMap prevents nesting Optional<Optional<B>>).
- Pattern:
  - optA.flatMap(a -> optB.map(b -> combine(a,b)))

4. Behavior summary
- Present case:
  - Optional.of(a).flatMap(f) == f(a)
- Empty case:
  - Optional.<A>empty().flatMap(f) == Optional.<B>empty()
- Short‑circuiting:
  - If any step is empty, the rest of the chain is skipped.

5. Sequencing dependent steps (typical shapes)
- Two steps:
  - oa.flatMap(a -> ob.map(b -> f(a,b)))
- Three steps:
  - oa.flatMap(a -> ob.flatMap(b -> oc.map(c -> g(a,b,c))))

6. Purity and referential transparency
- If the functions passed to flatMap/map are pure:
  - Same inputs → same outputs; no side effects.
- Side effects run only when prior Optionals are present.

7. Edge cases and tips
- Nulls:
  - Avoid Optional.of(null); use Optional.ofNullable.
- Readability:
  - For multiple steps, prefer extracting small methods or using intermediate variables for clarity.

8. Monad laws (for predictable refactoring)
- Left identity:  Optional.of(a).flatMap(f) == f(a)
- Right identity: opt.flatMap(Optional::of) == opt
- Associativity:  opt.flatMap(f).flatMap(g) == opt.flatMap(a -> f(a).flatMap(g))
  - These hold with pure functions and let you safely rearrange chains.
*/

  // 9) Natural Transformation: List<A> -> Optional<A> (head)
  static <A> Optional<A> headOption(List<A> xs) { return xs.isEmpty() ? Optional.empty() : Optional.of(xs.get(0)); }
/*
1. Line: a natural transformation between containers (List -> Optional)

- Code: headOption(xs)
- Meaning:
  - Converts a List<A> into an Optional<A> by taking its first element if it exists.
  - If the list is empty, returns Optional.empty(); otherwise, wraps the first element in Optional.of(...).

- Result examples:
  - headOption(List.of(10, 20))  → Optional[10]
  - headOption(List.of())        → Optional.empty

2. What “Natural Transformation” means (informal)
- A uniform, structure‑preserving mapping between type constructors F and G:
  - For all A, a function nat: F<A> -> G<A> that doesn’t “look inside” A or depend on A’s specifics.
- Here:
  - F = List, G = Optional, nat = headOption
  - Uniform: same logic for any A (integers, strings, custom types).

3. Why this is useful
- Changes context, not the value type:
  - Collapses “many (possibly zero)” (List) into “zero or one” (Optional).
- Enables composing APIs that expect Optional without reshaping element types.

4. Behavior and properties
- Total function for any List<A>:
  - Empty list → Optional.empty()
  - Non‑empty list → Optional.of(first element)
- Pure and referentially transparent:
  - Same input list → same output Optional; no side effects.

5. Edge cases and tips
- Null elements:
  - If xs.get(0) might be null, wrap via Optional.ofNullable(xs.get(0)) to avoid NPE.
- Performance:
  - O(1) time/O(1) space; only checks emptiness and possibly reads index 0.

6. Variations
- headOption with streams/iterables:
  - For Stream<A>: xs.findFirst()
- lastOption:
  - Return the last element instead of the first (O(1) for random access lists, O(n) for linked lists).
- safeHead for arrays:
  - array.length == 0 ? Optional.empty() : Optional.of(array[0])

7. Composition intuition
- Natural transformations compose:
  - If you had another transformation Optional<A> -> Either<E, A>, composing it with headOption gives List<A> -> Either<E, A>, still uniform and type‑parametric.
*/

  // 10) Monoid with reduce (associativity + identity)
  static final int mSum = List.of(1,2,3).stream().reduce(0, Integer::sum);
  static final String mStr = List.of("a","b","c").stream().reduce("", String::concat);
/*
1. Line: reduce with a Monoid (operation + identity)

- Code:
  - reduce(0, Integer::sum)
  - reduce("", String::concat)
- Meaning:
  - A Monoid is a pair (⊕, e) with:
    - Associative binary operation ⊕: (x ⊕ y) ⊕ z == x ⊕ (y ⊕ z)
    - Identity element e: e ⊕ x == x == x ⊕ e
  - For ints under addition: (⊕ = +, e = 0)
  - For strings under concatenation: (⊕ = concat, e = "")

- Why reduce needs a Monoid:
  - Folding a collection safely requires an operation that can combine elements in any grouping (associativity) and a neutral start value (identity).

2. Behavior (step-by-step)

- mSum:
  - Start acc = 0
  - 0 + 1 = 1
  - 1 + 2 = 3
  - 3 + 3 = 6
  - Result: 6

- mStr:
  - Start acc = ""
  - "" concat "a" = "a"
  - "a" concat "b" = "ab"
  - "ab" concat "c" = "abc"
  - Result: "abc"

3. Why associativity and identity matter

- Associativity:
  - Streams and parallel reductions may regroup operations; associativity guarantees the same result.
  - Example: (("a" + "b") + "c") == ("a" + ("b" + "c")) == "abc"

- Identity:
  - The seed value must be neutral so it doesn’t affect the outcome:
    - 0 + x = x
    - "" + s = s

4. Purity and determinism

- With pure operations (Integer::sum, String::concat), reduce is pure and referentially transparent:
  - Same input → same output; no side effects.

5. Edge cases and tips

- Empty input:
  - reduce(identity, op) returns the identity (0 or "")
- Non-associative ops:
  - Avoid using non-associative operations (e.g., subtraction, floating‑point addition with high sensitivity to order) for general reduce, especially in parallel contexts.
- Performance:
  - String::concat in a loop can be O(n^2) due to intermediate strings; prefer StringBuilder for large concatenations or use Collectors.joining for streams.

6. Custom Monoids (examples)

- Product monoid (ints):
  - op = (a, b) -> a * b, identity = 1
- Max/Min monoids:
  - op = Math::max with identity = Integer.MIN_VALUE (careful in empty cases)
  - op = Math::min with identity = Integer.MAX_VALUE

7. Parallel friendliness

- Associativity enables parallel reduction:
  - stream().parallel().reduce(identity, op) remains correct for true monoids.
*/

  // 11) ADTs & Pattern Matching (sealed + switch)
  sealed interface Shape permits Circle, Rect {}
  record Circle(double r) implements Shape {}
  record Rect(double w, double h) implements Shape {}
  static double area(Shape s) {
    return switch (s) {
      case Circle c -> Math.PI * c.r() * c.r();
      case Rect   r -> r.w() * r.h();
    };
  }
/*
1. Lines 1–3: define an Algebraic Data Type (ADT) with a sealed hierarchy

- Code:
  - sealed interface Shape permits Circle, Rect {}
  - record Circle(double r) implements Shape {}
  - record Rect(double w, double h) implements Shape {}
- Meaning:
  - Sealed interface restricts which types can implement it (only Circle and Rect).
  - This creates a closed set of variants (a sum type): Shape = Circle | Rect.
  - Records are immutable data carriers with generated constructor/accessors/equals/hashCode/toString.

- Why sealed matters:
  - The compiler knows all possible cases of Shape, enabling exhaustive pattern checks in switch.
  - Prevents accidental extension from outside the declaring compilation unit.

2. Line 5: pattern matching with switch (exhaustive over a sealed type)

- Code:
  - switch (s) { case Circle c -> ...; case Rect r -> ...; }
- Meaning:
  - Deconstructs s by its concrete variant and binds fields (c.r(), r.w(), r.h()).
  - For sealed types, the compiler enforces exhaustiveness (all known cases handled).
  - No default branch needed; adding a new variant forces you to update this switch.

3. Behavior (step-by-step)

- If s is Circle(r):
  - area = π * r * r
- If s is Rect(w, h):
  - area = w * h

4. Why ADTs + pattern matching are powerful

- Clarity:
  - Domain models with a finite set of cases are explicit and self‑documenting.
- Safety:
  - Exhaustive matching prevents “forgotten cases” at compile time.
- Immutability:
  - Records’ fields are final; values are stable and thread‑safe by default.
- Refactoring:
  - Adding a new Shape variant triggers compile errors at all pattern matches until you handle it.

5. Purity and reasoning

- area is pure:
  - No mutation, no I/O; same input Shape gives same numeric result.
- Referential transparency:
  - You can replace area(Circle(2)) with its numeric value without changing behavior.

6. Edge cases and tips

- Negative/NaN parameters:
  - The model does not validate; consider guards/constructors if needed.
- Extensibility:
  - To add Triangle, update permits list and all switch expressions—compiler guides the changes.
- Performance:
  - Pattern matching is O(1) dispatch; records are lightweight value types (by reference).
*/


  // 12) Effects at the Edges — pure core + I/O boundary
  static int pureLogic(int x) { return x * 2; }

  public static void main(String[] args) {
    System.out.println(pureLogic(5)); // effect at boundary
  }
/*
1. Lines: separate pure core from I/O

- Code:
  - pureLogic: a pure function (no I/O, no mutation, deterministic).
  - main: performs I/O (printing) at the program boundary, calling the pure core.

- Meaning:
  - Keep computation (business logic) pure.
  - Perform side effects (I/O) only at edges (main, controllers, handlers, etc.).

2. Why this separation matters

- Testability:
  - pureLogic is trivial to unit test: same input → same output.
- Reasoning & refactoring:
  - Pure code is referentially transparent and easier to refactor safely.
- Reuse & composition:
  - Pure functions compose; you can pipeline them without worrying about hidden effects.
- Reliability & observability:
  - Centralize side effects for retries, logging, metrics, and error handling.

3. Behavior (step-by-step)

- pureLogic(5) → 10 (no effects).
- main calls System.out.println(10) → prints "10\n" (effect).

4. Purity vs effects

- pureLogic:
  - No global state, no I/O, no time randomness; returns x * 2.
- println:
  - Writes to stdout (side effect), not referentially transparent.

5. Edge cases and tips

- Exceptions:
  - Catch and handle around I/O boundaries; keep core pure.
- Time, randomness, config:
  - Pass them in as parameters (or abstract behind interfaces) rather than reading directly inside pure functions.
- Nulls:
  - Prefer Optional/validation types at the edges; keep core domain types non-null.

6. Testing strategy

- Unit tests:
  - assertEquals(10, pureLogic(5));
- Integration/acceptance:
  - Verify console/log output or injected writer at the boundary.

7. Refactoring patterns

- Dependency injection at the edge:
  - Pass an Output (PrintStream/Writer/Logger) into main/runner; core returns values only.
- Functional style:
  - Core returns data; the edge decides how to effect (print, save, send, etc.).
*/

  // 13) Property‑Based Testing (outline with jqwik)
//   @Property
//   void functorIdentity(@ForAll List<Integer> xs){
//     Assertions.assertEquals(xs, xs.stream().map(Function.identity()).toList());
//   }
/*
1. Line: a property, not an example — test many random inputs

- Code:
  - @Property
  - functorIdentity(@ForAll List<Integer> xs) { ... }
- Meaning:
  - Property-based test: the framework generates many random xs lists.
  - The assertion must hold for all generated inputs, not just a few hand-picked cases.

2. The property being checked (Functor identity law for map)

- Law:
  - xs.map(identity) == xs
- In Java streams:
  - xs.stream().map(Function.identity()).toList() should equal xs
- Intuition:
  - Mapping with the identity function should not change the structure or the values.

3. Why property‑based tests are powerful

- Broad input coverage:
  - Randomized data explores edge cases (empty list, large lists, duplicates, negatives).
- Less bias:
  - You specify the law; the tool probes many cases automatically.
- Shrinking (when supported):
  - On failure, the framework reduces the input to a minimal counterexample, easing debugging.

4. Determinism and purity

- To make failures reproducible:
  - Tests should be deterministic for a given seed; avoid non‑deterministic or effectful code inside the property.
- Pure transformations (like identity/map) are ideal for property checks.

5. Extending the idea (other fundamental laws)

- Functor composition:
  - xs.map(x -> g(f(x))) == xs.map(f).stream().map(g).toList()
- Monoid laws (with reduce):
  - Identity: reduce(e, ⊕) on empty equals e
  - Associativity: fold groupings don’t change the result
- Option/Optional laws:
  - map(identity) == self
  - map(f).map(g) == map(x -> g(f(x)))

6. Edge cases to consider (help the generator or assert assumptions)

- Nulls:
  - If inputs may contain nulls, decide whether to filter or handle explicitly.
- Very large inputs:
  - Ensure performance is acceptable; consider size limits.
- Equality semantics:
  - Use deep equality (list order and contents) for structural comparisons.

7. Minimal runnable shape (pseudocode with annotations)

- @Property
- void functorIdentity(@ForAll List<Integer> xs) {
-   var mapped = xs.stream().map(Function.identity()).toList();
-   Assertions.assertEquals(xs, mapped);
- }

8. What a failure would mean

- If this property fails:
  - Either map(identity) changed the list (unexpected), or equality/collection conversion isn’t comparing what you think.
  - Inspect the failing input (and the shrunk case) to pinpoint the issue (e.g., custom equals, mutation, or incorrect mapping step).
*/
}